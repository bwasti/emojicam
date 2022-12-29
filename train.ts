import { Image } from '@shumai/image'
import * as sm from '@shumai/shumai'
import { readdir } from 'node:fs/promises'
import * as path from 'path'
import { EmojiClassifier } from './model'

const dataset = []
const unicodes = []

// number of random mutations of each image
const mut_per_img = 32

function indexToEmoji(index) {
  const unicode = unicodes[index]
  const emoji = unicode
    .split('-')
    .map((codePoint) => String.fromCodePoint(`0x${codePoint}`))
    .join('')
  return emoji
}

const files = await readdir('emojis')

console.log('loading images')
let emoji = ''
const base_imgs = []
for (const file of sm.util.viter(files, () => emoji)) {
  const p = path.join('emojis', file)
  let base_img = new Image(p)
  // ignore the flat emoji for now
  if (base_img.channels !== 4) {
    continue
  }
  base_img = base_img.resize(0.5)
  const components = file.split('.')[0].split('_')
  if (components.length === 2) {
    unicodes.push(components[1])
  } else {
    unicodes.push(components[2])
  }
  emoji = indexToEmoji(unicodes.length - 1)
  base_imgs.push(base_img)
}

function mutateColor(img) {
  let t = img.tensor().astype(sm.dtype.Float32)
  const r = 1 + (Math.random() - 0.5) / 50
  const g = 1 + (Math.random() - 0.5) / 50
  const b = 1 + (Math.random() - 0.5) / 50
  const s = 1 + (Math.random() - 0.5) / 5
  t = t
    .div(sm.scalar(255))
    .mul(sm.tensor(new Float32Array([r, g, b, 1])))
    .mul(sm.scalar(s))
    .mul(sm.scalar(255))
  t = sm.clamp(t, 0, 255)
  return new Image(t)
}

function randomCrop(img) {
  // max 20% shift
  const scale = 1 + Math.random() / 5
  const offset_x = Math.floor(Math.random() * img.width * (scale - 1))
  const offset_y = Math.floor(Math.random() * img.height * (scale - 1))
  return img.resize(scale).crop(offset_x, offset_y, img.width, img.height)
}

const emojilist = []
for (const i of sm.util.range(unicodes.length)) {
  emojilist.push(indexToEmoji(i))
}
Bun.write('emoji.json', JSON.stringify(emojilist))

console.log('mutating images')
let index = 0
for (const base_img of sm.util.viter(base_imgs, () => emoji)) {
  const h = base_img.height
  emoji = indexToEmoji(index)

  for (const i of sm.util.range(mut_per_img)) {
    let img = base_img.rotate((Math.random() - 0.5) * 20)
    const scale = h / img.height
    img = img.resize(scale)
    img = mutateColor(img)
    if (Math.random() > 0.5) {
      img = img.gaussblur(Math.random() / 2)
    }

    img = randomCrop(img)
    img = img.flatten(30, 30, 30)
    const t = img.tensor().transpose([2, 0, 1]).astype(sm.dtype.Float32).div(sm.scalar(255))
    const ohe = sm.full([unicodes.length], 0).indexedAssign(sm.scalar(1), [index])
    dataset.push([t.unsqueeze(0), ohe.unsqueeze(0)])
  }
  index += 1
}

console.log('training model')
const model = new EmojiClassifier(unicodes.length)
model.checkpoint('weights', (i) => i % 100 === 0)

function getBatch(ds) {
  if (ds.length === 0) {
    const idx = Math.floor(Math.random() * dataset.length)
    return dataset[idx]
  }
  const Xs = []
  const Ys = []
  for (const i of sm.util.range(ds[0])) {
    const [X, Y] = getBatch(ds.slice(1))
    Xs.push(X)
    Ys.push(Y)
  }
  return [sm.concat(Xs, 0), sm.concat(Ys, 0)]
}

let ema_loss = 0
let ema_accuracy = 0
let example = ''
const loss_fn = sm.loss.mse
const optim = new sm.optim.Adam(1e-3)

function test() {
  const [X, Y] = getBatch([2, 4, 4])
  const Y_hat = model(X)
  const tes = sm.argmax(Y_hat, 1)
  const ref = sm.argmax(Y, 1)
  const acc = (100 * tes.eq(ref).sum().toFloat32()) / Y.shape[0]
  const example = [
    indexToEmoji(ref.toInt32()),
    unicodes[ref.toInt32()],
    indexToEmoji(tes.toInt32()),
    unicodes[tes.toInt32()]
  ]
  return [acc, example]
}

function dump(i) {
  if (i % 100 === 0) {
    const [accuracy, ex] = test()
    if (ema_accuracy === 0) {
      ema_accuracy = accuracy
    }
    ema_accuracy = 0.9 * ema_accuracy + 0.1 * accuracy
    example = `${ex[0]} (${ex[1]}) → ${ex[2]} (${ex[3]})`
  }
  return `loss: ${ema_loss.toFixed(4)}, acc: ${ema_accuracy.toFixed(2)}%, ex: ${example}`
}

for (const iter of sm.util.viter(50000, dump)) {
  const [X, Y] = getBatch([2, 4, 4])
  const Y_hat = model(X)
  const loss = loss_fn(Y_hat, Y)
  if (ema_loss === 0) {
    ema_loss = loss.toFloat32()
  }
  ema_loss = 0.9 * ema_loss + 0.1 * loss.toFloat32()
  const grads = loss.backward()
  optim.step(grads)
}
