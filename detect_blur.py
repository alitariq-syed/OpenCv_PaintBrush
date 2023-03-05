   
    
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import blur_detector


target_size = (640,640)
imgs_folder = "./blurred"
out_folder = "./blur_masks"



# def detect_blur(img_path):

   
#     img_mask = np.zeros(target_size, np.uint8)
#     img = cv2.imread(imgs_folder+'/'+img_path)
#     print(img.shape)
#     if img.shape[0:2]!=target_size:
#         img_resized = cv2.resize(img,target_size,interpolation=cv2.INTER_AREA)
#     else:
#         img_resized = img
#     print(img_resized.shape)
    
#     # plt.imshow(img_resized)
#     cv2.namedWindow('image')
#     cv2.imshow('image',img_resized)
    
#     gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
#     plt.imshow(gray,'gray')
    
#     lp = cv2.Laplacian(gray, cv2.CV_64F)
#     plt.imshow(lp,'gray'),plt.show()
#     lp=cv2.normalize(lp,lp,0,255,cv2.NORM_MINMAX)
#     plt.imshow(lp,'gray'),plt.show()
#     lp=lp.astype(np.uint8)
#     plt.imshow(lp,'gray'),plt.show()
    
    
    
#     var = cv2.Laplacian(gray, cv2.CV_64F)
#     # grad_mag
    
#     edges_x = cv2.filter2D(img,cv2.CV_32F,kernelx)#gradient x
#     edges_y = cv2.filter2D(img,cv2.CV_32F,kernely)#gradient y
#     grad=np.sqrt(np.square(edges_x)+np.square(edges_y))#gradient magnitude
    
#     edges_x=cv2.normalize(edges_x,edges_x,0,255,cv2.NORM_MINMAX)#normalize
#     # edges_y=cv2.normalize(edges_y,edges_y,0,255,cv2.NORM_MINMAX)#normalize
#     grad=cv2.normalize(grad,grad,0,255,cv2.NORM_MINMAX)#normalize
    
#     edges_x=edges_x.astype(np.uint8)
#     edges_y=edges_y.astype(np.uint8)
#     grad=grad.astype(np.uint8)

#     cv2.imshow('Gradients_X',edges_x)
#     cv2.imshow('Gradients_Y',edges_y)
#     cv2.imshow('Gradients',grad)




#     # cv2.imwrite(masks_folder+'/'+img_path[:-4]+"_mask.png",img_mask)
#     cv2.waitKey(0) # waits until a key is pressed
#     cv2.destroyAllWindows() # destroys the window showing image

# def main():
for file in os.listdir(imgs_folder):
    # file = "8399166846_f6fb4e4b8e_k.png"
    img = cv2.imread(imgs_folder+'/'+file, 0)
    plt.imshow(img),plt.show()
    blur_map = blur_detector.detectBlur(img, downsampling_factor=8, num_scales=4, scale_start=2, num_iterations_RF_filter=3)
    blur_map = (blur_map*255).astype(np.uint8)
    plt.imshow(blur_map),plt.show()
    # cv2.imshow('ori_img', img)
    # cv2.imshow('blur_map', blur_map)

    # Otsu's thresholding
    ret2,th2 = cv2.threshold(blur_map,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # Otsu's thresholding after Gaussian filtering
    blur = cv2.GaussianBlur(blur_map,(5,5),0)
    ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # plot all the images and their histograms
    plt.imshow(th2),plt.show()
    plt.imshow(th3),plt.show()


    cv2.waitKey(0)

# if __name__ == "__main__":
#     main()