import { Image } from '@shumai/image'
import { readdir } from 'node:fs/promises'
import * as path from 'path'
import * as sm from '@shumai/shumai'
import { EmojiClassifier } from './model'
import { serve, file } from "bun";

const emoji = JSON.parse(Bun.readFile('emoji.json'))
const model = new EmojiClassifier(emoji.length)
model.checkpoint('weights') // this loads all the weights

serve({
  async fetch(req) {
    if (req.method === 'POST') {
      const response = [] 
      const data = new Uint8Array(await req.arrayBuffer())
      
      // convert data to a float tensor
      let t = sm.tensor(data).astype(sm.dtype.Float32)

      // mask out the alpha channel
      t = t.reshape([36 * 36, 48 * 36, 4]).eval().index([':', ':', ':3'])

      // arrayfire only supports dim <= 4 :( so we have to hack
      t = t.reshape([36, 36, 48, 36 * 3])
      t = t.transpose([0, 2, 1, 3])
      t = t.reshape([36 * 48, 36, 36, 3])
      t = t.transpose([0, 3, 1, 2]).div(sm.scalar(255))

      // run the model
      const out = model(t).argmax(1)

      // convert indices back to emoji (strings)
      for (let i of out.toInt32Array()) {
        response.push(emoji[i])
      }
      
      // all set to return!
      return new Response(JSON.stringify(response))
    }
    return new Response(file("./index.html"))
  },
})
