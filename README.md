# img2emoji

Code for training

![trainloop](https://user-images.githubusercontent.com/4842908/209762478-f233e811-fb4f-48c4-8e67-4d8d51bebac0.gif)

and serving

![img2emoji_smallest](https://user-images.githubusercontent.com/4842908/209761575-1a3d1bce-c8d1-49ee-9836-89c5a4a3e759.gif)

## What is this?

A logical extension to real-time ASCII webcams (e.g. [p2pvc](https://github.com/mofarrell/p2pvc)).

Since there are ~2k emoji used for this, the structure and color of each one is prohibitively annoying to encode in a lookup table. Instead, we use a tiny neural network to infer which emoji best represents any given 36x36 patch of pixels.  Training involves mutating reference images of emoji (with libvips) and having the neural network learn to predict the original emoji index.

At serving time, the neural network is called with a batch of 1,728 patches (36x48) around 5-10 times per second.

## Run

(Be sure to [install `bun`](https://bun.sh))

```
$ git clone https://github.com/bwasti/img2emoji.git
$ cd img2emoji
$ bun add @shumai/shumai
$ bun serve.ts
```

Then open `localhost:3000`

## Train from scratch

```
$ bun add @shumai/image
$ bun train.ts
```
