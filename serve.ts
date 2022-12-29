import { Image } from '@shumai/image'
import * as sm from '@shumai/shumai'
import { file, serve } from 'bun'
import { readdir } from 'node:fs/promises'
import * as path from 'path'
import { EmojiClassifier } from './model'

const emoji = JSON.parse(Bun.readFile('emoji.json'))
const model = new EmojiClassifier(emoji.length)
model.skip_softmax = true
model.checkpoint('weights') // this loads all the weights

serve({
  async fetch(req) {
    if (req.method === 'POST') {
      const response = []
      const data = new Uint8Array(await req.arrayBuffer())

      const t0 = performance.now()

      sm.util.tidy(() => {
        // convert data to a float tensor
        let t = sm.tensor(data).astype(sm.dtype.Float32)
        const height = 54
        const width = 72

        // mask out the alpha channel
        t = t.reshape([height * 36, width * 36, 4]).index([':', ':', ':3'])

        // arrayfire only supports dim <= 4 :( so we have to hack
        t = t.reshape([height, 36, width, 36 * 3])
        t = t.transpose([0, 2, 1, 3])
        t = t.reshape([height * width, 36, 36, 3])
        t = t.transpose([0, 3, 1, 2]).div(sm.scalar(255)).eval()

        // run the model
        const out = model(t).argmax(1)

        // convert indices back to emoji (strings)
        for (const i of out.toInt32Array()) {
          response.push(emoji[i])
        }
      })

      const t1 = performance.now()
      sm.util.tuiLoad(`${1e3 / (t1 - t0)} iters/sec`)

      // all set to return!
      return new Response(JSON.stringify(response))
    }
    return new Response(file('./index.html'))
  }
})
