# img2emoji

Repo for training

![trainloop](https://user-images.githubusercontent.com/4842908/209762478-f233e811-fb4f-48c4-8e67-4d8d51bebac0.gif)

and serving in real time

![img2emoji_smallest](https://user-images.githubusercontent.com/4842908/209761575-1a3d1bce-c8d1-49ee-9836-89c5a4a3e759.gif)


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
