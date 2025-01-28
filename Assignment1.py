import cv2
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

#use class to avoid using global variables
class PlotImage():
    def __init__(self):
        """
        Function to initialize class attributes.
        """
        self.DIR = r'dog.bmp'
        self.ORIGINAL_IMG = cv2.imread(self.DIR, cv2.IMREAD_GRAYSCALE)
        self.img = self.ORIGINAL_IMG.copy()
        self.adjusted_img = None
    
    def save_image(self, event):
        """
        Function to save file on button press.

        Args: 
        event: Button click event
        """
        self.adjusted_img = self.update_image(None)
        cv2.imwrite(self.DIR, self.adjusted_img)

    def update_image(self,val):
        """
        Function to update images and histograms.

        Args:
        val: slider value

        Returns:
        self.adjusted_img: Modified image after contrast and brightness update
        """
        #get slider vals
        contrast = contrast_slider.val
        brightness = brightness_slider.val
        
        #save to adjusted image variable
        self.adjusted_img = cv2.convertScaleAbs(self.img, alpha=contrast, beta=brightness)
        
        #update image
        ax_img2.clear()
        ax_img2.imshow(self.adjusted_img, 'gray')
        ax_img2.set_title("Modified Image")
        ax_img2.axis('off')
        
        #update historgram
        ax_hist2.clear()
        ax_hist2.set_title("Modified Histogram")
        
        hist = cv2.calcHist([self.adjusted_img], [0], None, [255], [0, 255])
        ax_hist2.plot(hist, color='black')
        ax_hist2.set_xlim([0, 255])
        
        #redraw plot
        fig.canvas.draw_idle()
        return self.adjusted_img

if __name__ == "__main__":
    myPlot = PlotImage()
    #initialize subplots and axes
    fig, axes = plt.subplots(2, 2, figsize=(12, 6))
    ax_img, ax_img2 = axes[0]
    ax_hist, ax_hist2 = axes[1]

    #calculate original histogram
    hist = cv2.calcHist([myPlot.img], [0], None, [255], [0, 255])

    #set image labels
    ax_img.imshow(myPlot.ORIGINAL_IMG, cmap='gray')
    ax_img.set_title("Original Image")
    ax_img.axis('off')
    ax_img2.imshow(myPlot.img, cmap='gray')
    ax_img2.set_title("Modified Image")
    ax_img2.axis('off')

    #set histogram labels
    ax_hist.set_xlabel("Intensity")
    ax_hist.set_ylabel("Frequencies")
    ax_hist.plot(hist, color = 'black')
    ax_hist.set_title("Original Histogram")
    ax_hist2.plot(hist, color = 'black')
    ax_hist2.set_title("Modified Histogram")

    #set slider pos and function calls
    ax_contrast = plt.axes([0.7, 0.4, 0.25, 0.03], facecolor='lightgray')
    ax_brightness = plt.axes([0.7, 0.6, 0.25, 0.03], facecolor='lightgray')
    contrast_slider = Slider(ax_contrast, '', 0.0, 3.0, valinit=1, valstep=0.1)
    brightness_slider = Slider(ax_brightness, '', -100, 100, valinit=0, valstep=1)
    contrast_slider.on_changed(myPlot.update_image)
    brightness_slider.on_changed(myPlot.update_image)

    #set save button pos and function calls
    ax_save_button = plt.axes([0.75, 0.25, 0.15, 0.03])
    save_button = Button(ax_save_button, 'Save')
    save_button.on_clicked(myPlot.save_image)

    plt.text(0.3, 13, "Brightness:")
    plt.text(0.35, 6.5, "Contrast:")
    #shift subplot over and show
    plt.subplots_adjust(right=0.65)
    plt.show()