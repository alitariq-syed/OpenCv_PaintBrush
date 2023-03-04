import cv2
import numpy as np
import os


drawing = False # true if mouse is pressed
ix,iy = -1,-1
target_size = (640,640)
imgs_folder = "./imgs"
masks_folder = "./masks"


def draw_mask(img_path):

    # mouse callback function
    def draw_circle(event,x,y,flags,param):
        global ix,iy,drawing
        

        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            ix,iy = x,y

        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing == True:
                cv2.circle(img_resized,(x,y),5,(0,0,255),-1)
                cv2.circle(img_mask,(x,y),5,(255),-1)

        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            cv2.circle(img_resized,(x,y),5,(0,0,255),-1)
            cv2.circle(img_mask,(x,y),5,(255),-1)

    img_mask = np.zeros(target_size, np.uint8)
    img = cv2.imread(imgs_folder+'/'+img_path)
    print(img.shape)
    if img.shape[0:2]!=target_size:
        img_resized = cv2.resize(img,target_size,interpolation=cv2.INTER_AREA)
        cv2.imwrite(masks_folder+'/'+img_path,img_resized)

    else:
        img_resized = img
    print(img_resized.shape)

    cv2.namedWindow('image')
    cv2.setMouseCallback('image',draw_circle)

    while(1):
        cv2.imshow('image',img_resized)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
    
    cv2.floodFill(img_mask, None, (0, 0), (255))
    img_mask = cv2.bitwise_not(img_mask)
    cv2.imwrite(masks_folder+'/'+img_path[:-4]+"_mask.png",img_mask)
    # cv2.imwrite("img.png",img_resized)
    
    
    cv2.destroyAllWindows()

def main():
    for file in os.listdir(imgs_folder):
        # file = "8399166846_f6fb4e4b8e_k.png"

        draw_mask(file)


if __name__ == "__main__":
    main()