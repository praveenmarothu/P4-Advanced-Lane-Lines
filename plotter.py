
import matplotlib.pyplot as plt

def plot_images( images ,rows , cols,file_name):

    for index in range(0,len(images)):
        plt.subplot(rows,cols,index+1)
        plt.title(images[index][1])
        plt.tick_params(axis='both', which='major', labelsize=8)
        plt.tick_params(axis='both', which='minor', labelsize=6)
        if(len(images[index])==2):
            plt.imshow(images[index][0])
        else:
            plt.imshow(images[index][0],images[index][2])
    plt.tight_layout()
    plt.savefig(file_name)
