import keyboard


def on_key_press(event):
    if event.name == 'b':
        print("The 'b' key is pressed.")


if __name__ == '__main__':
    while True:
        keyboard.on_press(on_key_press)
        keyboard.wait()
