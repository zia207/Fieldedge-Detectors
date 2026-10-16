from PIL import Image

def cut(id,vx,vy):

    name1 = "C:\\Users\\edmun\\Documents\\ArcGIS\\AddIns\\"+id+".tif"
    name2 = "C:\\Users\\edmun\\Documents\\ArcGIS\\AddIns\\crop\\"+ id +"_crop_"
    im =Image.open(name1)
    #im=cv.imread(name1)

    dx =50
    dy = 50
    n = 1

    x1 = 0
    y1 = 0
    x2 = vx
    y2 = vy


    while x2 <= 1889:

        while y2 <= 1647:
            name3 = name2 + str(n) + ".tif"

            im2 = im.crop((y1, x1, y2, x2))
            im2.save(name3)
            y1 = y1 + dy
            y2 = y1 + vy
            n = n + 1
        x1 = x1 + dx
        x2 = x1 + vx
        y1 = 0
        y2 = vy

    print("crop image value:")
    return n-1


if __name__=="__main__":

    id = "new1"

    res = cut(id,192,192)

    print(res)