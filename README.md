# emojicam

An emoji filter for your webcam!  *([writeup](https://jott.live/markdown/images_as_emoji))*

![render](https://user-images.githubusercontent.com/4842908/209887791-65fd8e66-e95a-4e46-95db-552972540903.gif)

For fidelity testing I used some colorful album art.  Recognize any of them?


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

## Some stills

<img width="854" alt="Screenshot 2022-12-28 at 7 49 36 PM" src="https://user-images.githubusercontent.com/4842908/209889797-f2749dad-9925-4283-b257-3229176c9c77.png">
<img width="854" alt="Screenshot 2022-12-28 at 7 48 24 PM" src="https://user-images.githubusercontent.com/4842908/209889801-15dda3d9-f404-47f7-9812-9c04482eaf60.png">
<img width="854" alt="Screenshot 2022-12-28 at 7 47 11 PM" src="https://user-images.githubusercontent.com/4842908/209889802-e87faee3-a99c-4c0a-9827-fb782e71636c.png">
<img width="854" alt="Screenshot 2022-12-28 at 7 46 40 PM" src="https://user-images.githubusercontent.com/4842908/209889804-dca40a2e-d6a4-458d-a883-8ca26a81a382.png">

