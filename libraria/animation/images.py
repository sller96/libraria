import matplotlib.animation as animation
import matplotlib.pyplot as plt

def animate_images(images):
    fig = plt.figure(figsize=(8, 8))

    im = plt.imshow(images[0])

    def animate_func(i):
        im.set_array(images[i])
        return [im]

    return animation.FuncAnimation(
        fig, animate_func, frames=range(len(images)), interval=1
    )
