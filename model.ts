import * as sm from "@shumai/shumai";

export class EmojiClassifier extends sm.module.Module {
  constructor(num_emojis) {
    super();
    this.conv0 = sm.module.conv2d(3, 8, 3, { stride: 2 });
    this.conv1 = sm.module.conv2d(8, 32, 3, { stride: 2 });
    this.linear = sm.module.linear(32, num_emojis);
  }
  forward(x) {
    x = sm.avgPool2d(x, 2, 2, 2, 2);
    x = this.conv0(x);
    x = x.tanh();
    x = sm.avgPool2d(x, 2, 2, 2, 2);
    x = this.conv1(x);
    x = x.tanh();
    x = x.reshape([x.shape[0], 32]);
    x = this.linear(x);
    return x.softmax(1);
  }
}
