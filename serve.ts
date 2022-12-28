import { Image } from '@shumai/image'
import { readdir } from 'node:fs/promises'
import * as path from 'path'
import * as sm from '@shumai/shumai'
import { EmojiClassifier } from './model'
import { serve, file } from "bun";

const emoji = JSON.parse(Bun.readFile('emoji.json'))
const model = new EmojiClassifier(files.length)
model.checkpoint('weights', () => false)

serve({
  async fetch(req) {
    if (req.method === 'POST') {
      const data = new Uint8Array(await req.arrayBuffer())
      let t = sm.tensor(data).reshape([2 * 648, 2 * 864, 4]).index([':', ':', ':3'])
      // arrayfire only supports dim <= 4 :( so we have to hack
      t = t.reshape([2 * 9, 72, 2 * 12, 72 * 3])
      t = t.transpose([0, 2, 1, 3])
      // only need 3 channels
      t = t.reshape([4 * 9 * 12, 72, 72, 3])
      t = t.transpose([0, 3, 1, 2])
      const out = model(t.astype(sm.dtype.Float32)).argmax(1)
      const response = [] 
      for (let i of out.toInt32Array()) {
        response.push(emoji[i])
      }
      return new Response(JSON.stringify(response))
    }
    return new Response(file("./index.html"))
  },
})
