from typing import Dict
from diffusion_policy.policy.common.base_policy import BasePolicy

class BaseImageRunner:
    def __init__(self, output_dir):
        self.output_dir = output_dir

    def run(self, policy: BasePolicy) -> Dict:
        raise NotImplementedError()

    def close(self):
        """释放 runner 持有的环境资源。

        无环境资源的 runner 可以沿用默认空实现；持有 vector env 的子类
        应覆盖本方法，并保证重复调用安全。
        """
        return None
