"""Use framework keyboard events while filtering OS key-repeat on toggles."""


def create_keyboard():
    from diffusion_policy.envs.realword.keystroke_counter import KeystrokeCounter

    class EdgeKeystrokeCounter(KeystrokeCounter):
        def __init__(self):
            self.pressed = set()
            super().__init__()

        def on_press(self, key):
            with self.lock:
                if key not in self.pressed:
                    self.pressed.add(key)
                    self.key_count_map[key] += 1
                    self.key_press_list.append(key)

        def on_release(self, key):
            with self.lock:
                self.pressed.discard(key)

    return EdgeKeystrokeCounter()
