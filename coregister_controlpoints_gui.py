import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sys

def load_images(img_1_path, img_2_path):
    img_1_arr = plt.imread(img_1_path)
    img_2_arr = plt.imread(img_2_path)

    return img_1_arr, img_2_arr

def save_image(img_1_arr, img_2_arr, M):

    img_2_arr_warped = cv2.warpPerspective(img_2_arr, M, (img_1_arr.shape[1], img_1_arr.shape[0]))
    out_filename = "static/out.jpeg"
    cv2.imwrite(out_filename, img_2_arr_warped[...,::-1])

    print("Registered Image Saved to " + out_filename)
    sys.exit()


def init_figs(img_1_arr,
              img_2_arr):

    # Create a figure with two subplots in a single row
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # convert that to uint8 for cv2
    plt.suptitle("Use the right mouse button to pick points; at least 4. \n"
                 "Close the figure when finished.")
    to_uint8 = lambda x: ((x - x.min()) / (x.max() - x.min()) * 255).astype(np.uint8)
    img_1_arr_uint8 = to_uint8(img_1_arr)
    img_2_arr_uint8 = to_uint8(img_2_arr)

    # Display the VNIR image on the left subplot
    ax1.imshow(img_1_arr_uint8)
    ax1.set_title('Image 1')

    # Display the SWIR image on the right subplot
    ax2.imshow(img_2_arr_uint8)
    ax2.set_title('Image 2 - To Be Warped')

    return fig, ax1, ax2, img_1_arr_uint8, img_2_arr_uint8

def main(img_1_path,
         img_2_path):
    global not_satisfied

    # load images
    img_1_arr, img_2_arr = load_images(img_1_path, img_2_path)


    not_satisfied = True
    while not_satisfied:
        fig, ax1, ax2, img_1_arr_uint8, img_2_arr_uint8 = init_figs(img_1_arr, img_2_arr)

        img1_points = []
        img2_points = []

        def on_click_vnir(event):
            if event.inaxes == ax1 and event.button == 3:  # Left mouse button clicked in VNIR subplot
                img1_points.append((event.xdata, event.ydata))
                ax1.plot(event.xdata, event.ydata, 'ro')  # Plot a red dot at the clicked point
                ax1.figure.canvas.draw_idle()

        def on_click_swir(event):
            if event.inaxes == ax2 and event.button == 3:  # Left mouse button clicked in SWIR subplot
                img2_points.append((event.xdata, event.ydata))
                ax2.plot(event.xdata, event.ydata, 'ro')  # Plot a red dot at the clicked point
                ax2.figure.canvas.draw_idle()

        # Connect the mouse click events to the respective axes
        fig.canvas.mpl_connect('button_press_event', on_click_vnir)
        fig.canvas.mpl_connect('button_press_event', on_click_swir)
        plt.show()


        print(f"Found point are:\n image [1]: {img1_points}\n image [2]: {img2_points}")
        # calculate homorgraphy based on points found
        # point passed to homography should be x, y order
        M, mask = cv2.findHomography(np.array(img2_points), np.array(img1_points), cv2.RANSAC, 5)


        # show the result and see if the use is satisfied
        fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(6,6))
        def on_key(event):
            global not_satisfied
            if event.key == 'escape':  # Close figure if Escape key is pressed
                not_satisfied = False;
                plt.close(fig)

        img2_warped_arr_temp = cv2.warpPerspective(img_2_arr, M, (img_1_arr.shape[1], img_1_arr.shape[0]))
        ax1.imshow(img_1_arr)
        ax1.set_title("Image [1]")
        ax2.imshow(img2_warped_arr_temp)
        ax2.set_title("Image [2] - warped")
        ax3.imshow(img2_warped_arr_temp,
                   alpha=0.5)
        ax3.imshow(img_1_arr,
                   alpha=0.5)
        ax3.set_title("Image [1 & 2] - overlapped")


        plt.suptitle('Overlay of Coregistered Image \n'
                     'if satisfied press Escape to save image\n'
                     'if NOT satisfied close the figure to restart.')
        fig.canvas.mpl_connect('key_press_event', on_key)
        plt.show()

    # save image at last
    save_image(img_1_arr, img_2_arr, M)


if __name__ == "__main__":
    img_1_path = "static/cup_1.jpeg"
    img_2_path = "static/cup_2.jpeg"

    main(img_1_path, img_2_path)

