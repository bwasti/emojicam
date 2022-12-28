# emojicam

An emoji filter for your webcam!  *([writeup](https://jott.live/markdown/images_as_emoji))*

![img2emoji_smallest](https://user-images.githubusercontent.com/4842908/209761575-1a3d1bce-c8d1-49ee-9836-89c5a4a3e759.gif)

## What is this?

A logical extension to real-time ASCII webcams (e.g. [p2pvc](https://www.gizmodo.com.au/2015/02/this-video-chat-software-runs-in-terminal-renders-your-face-in-ascii/)) trained and served entirely in TypeScript.

Since there are ~2k emoji used for this project, the structure and color of each one would be prohibitively annoying to encode in a lookup table. Instead, it uses a tiny neural network to infer which emoji best represents any given 36x36 patch of pixels.  Training involves mutating reference images of emoji (with libvips) and having the neural network learn to predict the original emoji index.  It's kind of a hack, but it seems to work fairly well.

At serving time, the neural network is called with a batch of 1,728 patches (36x48) around 5-10 times per second.

## Run

(Be sure to [install `bun`](https://bun.sh))

```
$ git clone https://github.com/bwasti/emojicam.git
$ cd emojicam
$ bun add @shumai/shumai
$ bun serve.ts
```

Then open `localhost:3000`

## Train from scratch

![train](https://user-images.githubusercontent.com/4842908/209850459-8bcfe735-d93e-4a83-82cc-0d817946c4b9.gif)


```
$ bun add @shumai/image
$ bun train.ts
```
