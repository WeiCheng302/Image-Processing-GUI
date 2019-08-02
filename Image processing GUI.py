import wx
import cv2
import os.path
import numpy as np
from matplotlib import pyplot as plt
import math

class WindowClass(wx.Frame):

    def __init__(self,parent,title):
        super(WindowClass,self).__init__(parent, title='Image Processing GUI',size=(800,600),style=wx.MAXIMIZE_BOX |
                 wx.MINIMIZE_BOX | wx.RESIZE_BORDER | wx.SYSTEM_MENU | wx.CAPTION | wx.CLOSE_BOX)
        # self.basicGUI()
        panel = wx.Panel(self, size=(800, 800))
        openfiledialog = wx.FileDialog(None, 'open', '', '', '(*.png)|*.png', wx.FD_OPEN | wx.FD_FILE_MUST_EXIST)
        if openfiledialog.ShowModal() == wx.ID_OK:
            path = openfiledialog.GetPath()
            print(path)
        w = os.path.split(path)[1]
        print(w)
        global img
        img = cv2.imread(w)
        # cv2.imshow('123',img)

    # Box1 image processing
        # Title
        titlebox1 = wx.StaticText(panel, -1, '1.Image Processing', (20, 20))
        titlebox1.SetForegroundColour('Black')

        # Buttons
        self.button11 = wx.Button(panel, -1, "1.1 Load Image", pos=(20, 40))
        self.Bind(wx.EVT_BUTTON, self.OnClick11, self.button11)
        self.button11.SetDefault()
        self.button12 = wx.Button(panel, -1, "1.2 Color Conversion", pos=(20, 70))
        self.Bind(wx.EVT_BUTTON, self.OnClick12, self.button12)
        self.button12.SetDefault()
        self.button13 = wx.Button(panel, -1, "1.3 Image Flipping", pos=(20, 100))
        self.Bind(wx.EVT_BUTTON, self.OnClick13, self.button13)
        self.button13.SetDefault()
        self.button14 = wx.Button(panel, -1, "1.4 Blending", pos=(20, 130))
        self.Bind(wx.EVT_BUTTON, self.OnClick14, self.button14)
        self.button14.SetDefault()

    # Box2 Edge Detection
        # Title
        titlebox2 = wx.StaticText(panel, -1, '2.Edge Detection', (20, 180))
        titlebox2.SetForegroundColour('Black')
        # Buttons
        self.button21 = wx.Button(panel, -1, "2.1 Filter", pos=(20, 200))
        self.button22 = wx.Button(panel, -1, "2.2 Edge", pos=(20, 230))
        self.button23 = wx.Button(panel, -1, "2.3 Magnitude ", pos = (20, 260))
        # self.button24 = wx.Button(panel, -1, "2.4 Direction ", pos = (20, 290))
        self.Bind(wx.EVT_BUTTON, self.OnClick21, self.button21)
        self.Bind(wx.EVT_BUTTON, self.OnClick22, self.button22)
        self.Bind(wx.EVT_BUTTON, self.OnClick23, self.button23)
        # self.Bind(wx.EVT_BUTTON, self.OnClick24, self.button24)
        self.button21.SetDefault()

    # Box3 Image Pyramids
        # Title
        titlebox3 = wx.StaticText(panel, -1, '3.Image Pyramids', (220, 20))
        titlebox3.SetForegroundColour('Black')
        # Buttons
        self.button31 = wx.Button(panel, -1, "3.1 Image Pyramids", pos=(220, 40))
        self.Bind(wx.EVT_BUTTON, self.OnClick31,    self.button31)
        self.button31.SetDefault()

    # Box4 Adaptive Threshold
        # Title
        titlebox4 = wx.StaticText(panel, -1, '4.Adaptive Threshold', (220, 90))
        titlebox4.SetForegroundColour('Black')
        # Buttons
        self.button41 = wx.Button(panel, -1, "4.1 Global Threshold", pos=(220, 110))
        self.Bind(wx.EVT_BUTTON, self.OnClick41, self.button41)
        self.button41.SetDefault()
        self.button42 = wx.Button(panel, -1, "4.2 Local Threshold", pos=(220, 140))
        self.Bind(wx.EVT_BUTTON, self.OnClick42, self.button42)
        self.button42.SetDefault()

    # Box5 Image Transformation
        # Title
        titlebox5 = wx.StaticText(panel, -1, '5.Image Transformation', (420, 20))
        titlebox5.SetForegroundColour('Black')
        titlebox51 = wx.StaticText(panel, -1, '5.1 Rot, Scale, Translate', (430, 40))
        titlebox51.SetForegroundColour('Black')
        titlebox511 = wx.StaticText(panel, -1, 'Parameters', (440, 60))
        titlebox511.SetForegroundColour('Black')
        # Buttons
        angle = wx.StaticText(panel, -1, 'Angle： ', (440, 80))
        angle.SetForegroundColour('Black'),
        self.angletext=wx.TextCtrl(panel,-1,'0', pos=(490, 80), size=(50, 20))
        scale = wx.StaticText(panel, -1, 'Scale： ', (440, 110))
        scale.SetForegroundColour('Black'),
        self.scaletext = wx.TextCtrl(panel,-1,'0', pos=(490, 110), size=(50, 20))
        tx = wx.StaticText(panel, -1, 'Tx： ', (440, 140))
        tx.SetForegroundColour('Black'),
        self.txtext = wx.TextCtrl(panel,-1,'0', pos=(490, 140), size=(50, 20))
        ty= wx.StaticText(panel, -1, 'Ty： ', (440, 170))
        ty.SetForegroundColour('Black'),
        self.tytext = wx.TextCtrl(panel,-1,'0', pos=(490, 170), size=(50, 20))
        self.button51 = wx.Button(panel, -1, "5.1 Rotation, Scaling, Translation", pos=(440, 200))
        self.Bind(wx.EVT_BUTTON, self.OnClick51, self.button51)
        self.button51.SetDefault()
        self.button52 = wx.Button(panel, -1, "5.2 Perspective Transform", pos=(440, 230))
        self.Bind(wx.EVT_BUTTON, self.OnClick52, self.button52)
        #self.button53 = wx.Button(panel, -1, "5.3 Perspective Transform", pos=(440, 260))
        #self.Bind(wx.EVT_BUTTON, self.OnClick53, self.button53)
        self.button52.SetDefault()

        self.Show(True)
        self.Center()

    def OnClick11(self, event):
        print(type(img))
        cv2.imshow('Loaded Image', img)
        w = img.shape
        k = str(w[0])
        p = str(w[1])
        wx.StaticText(self, -1, 'Image Width', (20, 300)),wx.StaticText(self, -1, k, (120, 300))
        wx.StaticText(self, -1, 'Image Height', (20, 320)),wx.StaticText(self, -1, p, (120, 320))
        # cv2.imshow('123', img)
        # staticBmp=wx.StaticBitmap(self,-1,img2,pos=(440,300),size=(100,100))
        cv2.waitKey(0)

    def OnClick12(self,event):
        [B, G, R] = cv2.split(img)
        chooseOneBox = wx.SingleChoiceDialog(None, 'Change color from RGB to...','Make Your Choice！',
                                             ['RGB', 'RBG', 'BRG','BGR','GRB','GBR','Gray'])
        if chooseOneBox.ShowModal() == wx.ID_OK:
            choice = chooseOneBox.GetStringSelection()
            if choice == 'RGB':
                display = cv2.merge([R,G,B])
            if choice == 'RBG':
                display = cv2.merge([R, B, G])
            if choice == 'BRG':
                display = cv2.merge([B, R, G])
            if choice == 'BGR':
                display = cv2.merge([B, G, R])
            if choice == 'GRB':
                display = cv2.merge([G, R, B])
            if choice == 'GBR':
                display = cv2.merge([G, B, R])
            if choice == 'Gray':
                display = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        cv2.imshow('Merged Image',display)
        cv2.waitkey(0)

    def OnClick13(self,event):
        chooseOneBox = wx.SingleChoiceDialog(None, 'Which axisto flip?', 'Make Your Choice！',
                                             ['X axis', 'Y axis', 'XY axis'])
        if chooseOneBox.ShowModal() == wx.ID_OK:
            global display
            choice = chooseOneBox.GetStringSelection()
            if choice == 'X axis':
                display = cv2.flip(img,int(0))
            if choice == 'Y axis':
                display = cv2.flip(img,  int(1))
            if choice == 'XY axis':
                display = cv2.flip(img, int(-1))
        global img13
        img13 = cv2.imshow('Flipped Image', display)
        img13
        cv2.waitkey(0)

    def OnClick14(self,event):
        def nothing(x):
            pass
        a = 0
        cv2.namedWindow('Blended Image')
        cv2.createTrackbar('Blending','Blended Image',0,100,nothing)
        switch = 'close '
        cv2.createTrackbar(switch, 'Blended Image', 0, 1, nothing)
        while(1):
            k = cv2.addWeighted( display, 0.01*a,img, 0.01*(100 - a), 0)
            cv2.imshow('Blended Image', k)
            cv2.waitKey(1)
            a = cv2.getTrackbarPos('Blending', 'Blended Image')
            s = cv2.getTrackbarPos(switch, 'Blended Image')
            if s == 1:
                break
            else:
                pass
        cv2.destroyAllWindows()

    def OnClick21(self,event):
        img2 = cv2.imread('screw.jpg',0)
        cv2.imshow('2',img2)
        print(type(img2))
        chooseOneBox = wx.SingleChoiceDialog(None, 'Which Filter ?', 'Make Your Choice！',
                                             ['Mean Filter', 'Gaussian Filter', 'Medium Filter','Bilateral Filter'])
        if chooseOneBox.ShowModal() == wx.ID_OK:
            global chosenFilter
            chosenFilter = chooseOneBox.GetStringSelection()
            global filteredImg

            if chosenFilter == 'Mean Filter':
                def nothing(x):
                    pass
                h = 1
                w = 1
                cv2.namedWindow('Mean Filter')
                cv2.createTrackbar('Filter Height', 'Mean Filter', 3, 11, nothing)
                cv2.createTrackbar('Filter Width', 'Mean Filter', 3, 11, nothing)
                switch = 'close '
                cv2.createTrackbar(switch, 'Mean Filter', 0, 1, nothing)
                while (1):
                    blur = np.array(cv2.blur(img2, (h,w)))
                    cv2.imshow('Mean Filter', blur)
                    cv2.waitKey(1)
                    h = int(cv2.getTrackbarPos('Filter Height', 'Mean Filter'))
                    w = int(cv2.getTrackbarPos('Filter Width', 'Mean Filter'))
                    filteredImg=blur
                    s = cv2.getTrackbarPos(switch, 'Mean Filter')
                    if s == 1:
                        break
                    else:
                        pass
                cv2.destroyAllWindows()

            if chosenFilter == 'Gaussian Filter':
                def nothing(x):
                    pass
                h=1
                w=1
                cv2.namedWindow('Gaussian Filter')
                cv2.createTrackbar('Filter Height', 'Gaussian Filter', 1, 11, nothing)
                cv2.createTrackbar('Filter Width', 'Gaussian Filter', 1, 11, nothing)
                switch = 'close '
                cv2.createTrackbar(switch, 'Gaussian Filter', 0, 1, nothing)
                while (1):
                    blur = cv2.GaussianBlur(img2, (h,w),0)
                    cv2.imshow('Gaussian Filter', blur)
                    cv2.waitKey(1)
                    h = int(cv2.getTrackbarPos('Filter Height', 'Gaussian Filter'))
                    if h%2==0:
                        h+=1
                    w = int(cv2.getTrackbarPos('Filter Width', 'Gaussian Filter'))
                    if w%2==0:
                        w+=1
                    filteredImg=blur
                    s = cv2.getTrackbarPos(switch, 'Gaussian Filter')
                    if s == 1:
                        break
                    else:
                        pass
                cv2.destroyAllWindows()

            if chosenFilter == 'Medium Filter':
                def nothing(x):
                    pass
                h=1
                cv2.namedWindow('Medium Filter')
                cv2.createTrackbar('Filter height', 'Medium Filter', 3, 11, nothing)
                switch = 'close '
                cv2.createTrackbar(switch, 'Medium Filter', 0, 1, nothing)
                while (1):
                    blur = cv2.GaussianBlur(img2, (h,h),0)
                    cv2.imshow('Medium Filter', blur)
                    cv2.waitKey(1)
                    h = int(cv2.getTrackbarPos('Filter height', 'Medium Filter'))
                    if h%2==0:
                        h+=1
                    filteredImg=blur
                    s = cv2.getTrackbarPos(switch, 'Medium Filter')
                    if s == 1:
                        break
                    else:
                        pass
                cv2.destroyAllWindows()

            if chosenFilter == 'Bilateral Filter':
                def nothing(x):
                    pass
                r=1
                sigma=1
                grsigma=1
                cv2.namedWindow('Bilateral Filter')
                cv2.createTrackbar('radian', 'Bilateral Filter', 3, 21, nothing)
                cv2.createTrackbar('sigma', 'Bilateral Filter', 3, 21, nothing)
                cv2.createTrackbar('gray scale sigma', 'Bilateral Filter', 3, 21, nothing)
                switch = 'close '
                cv2.createTrackbar(switch, 'Bilateral Filter', 0, 1, nothing)
                while (1):
                    blur = cv2.bilateralFilter(img2,r,sigma,grsigma)
                    cv2.imshow('Bilateral Filter', blur)
                    cv2.waitKey(1)
                    r = int(cv2.getTrackbarPos('Filter height', 'Bilateral Filter'))
                    sigma = int(cv2.getTrackbarPos('sigma', 'Bilateral Filter'))
                    grsigma = int(cv2.getTrackbarPos('gray scale sigma', 'Bilateral Filter'))
                    filteredImg=blur
                    s = cv2.getTrackbarPos(switch, 'Bilateral Filter')
                    if s == 1:
                        break
                    else:
                        pass
                cv2.destroyAllWindows()

    def OnClick22(self,event):
        img2 = cv2.imread('screw.jpg', 0)
        Gx=np.array([[-1, 0, 1],
                     [-2, 0, 2],
                     [-1, 0, 1]])
        Gy=np.array([[-1, -2, -1],
                     [0, 0, 0],
                     [1, 2, 1]])
        global sobelx
        global sobely
        sobelx=cv2.filter2D(img2, -1, Gx)
        sobely=cv2.filter2D(img2, -1, Gy)
        Sx=cv2.imshow('Sobel x', sobelx)
        Sx
        Sy=cv2.imshow('Sobel y', sobely)
        Sy
        smallest=np.amin(sobelx)
        biggest=np.amax(sobelx)
        size=sobelx.shape
        for i in range(size[0]):
            for j in range(size[1]):
                sobelx[i, j] = sobelx[i, j]*256/(biggest-smallest)

        smallest1 = np.amin(sobely)
        biggest1 = np.amax(sobely)
        size = sobely.shape
        for i in range(size[0]):
            for j in range(size[1]):
                sobely[i, j] = sobely[i, j] * 256 / (biggest1 - smallest1)

    def OnClick23(self, event): #fail to use track bar
        sobel=cv2.add(sobelx, sobely)
        #a = 100
        def nothing(x):
            pass
        cv2.namedWindow('Magnitude')
        cv2.imshow('Magnitude', sobel)
        #cv2.createTrackbar('Magnitude', 'Magnitude', 0, 255, nothing)
        switch = 'close '
        cv2.createTrackbar(switch, 'Magnitude', 0, 1, nothing)
        while(1):
            #k= cv2.threshold(sobel,127,255,cv2.THRESH_BINARY)
            k=cv2.adaptiveThreshold(sobel,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY,11,2)
            cv2.imshow('Magnitude', k)
            cv2.waitKey(1)
            #a = cv2.getTrackbarPos('Magnitude', 'Magnitude')
            s = cv2.getTrackbarPos(switch, 'Magnitude')
            if s == 1:
                break
            else:
                pass
        cv2.destroyAllWindows()

    def OnClick31(self, event):
        a=cv2.imread('pyramid.jpg', 0)
        aa=cv2.GaussianBlur(a, (3, 3), 0)
        size=a.shape
        # k=a.copy()
        Gaussian = cv2.pyrDown(aa)
        Laplace = cv2.pyrUp(aa)
        q = cv2.resize(Gaussian, (size[0], size[1]))
        p = cv2.resize(Laplace, (size[0], size[1]))
        invG =a-q
        invL = a-p
        cv2.imshow('Gaussian', Gaussian)
        cv2.imshow('Laplace', Laplace)
        cv2.imshow('Inverse Gaussian', invG)
        cv2.imshow('Inverse Laplace', invL)

    def OnClick41(self, event):
        qr = cv2.imread('QR.png', 0)
        ret, qrresult = cv2.threshold(qr, 80, 0, cv2.THRESH_BINARY)
        cv2.imshow('QR', qr)
        cv2.imshow('QR result', qrresult)

    def OnClick42(self, event):
        qr = cv2.imread('QR.png', 0)
        qrresult = cv2.adaptiveThreshold(qr, 19, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                                       cv2.THRESH_BINARY, 11, 2)
        cv2.imshow('QR result', qrresult)

    def OnClick51(self, event):
        print(self.txtext.Value)
        cv2.namedWindow('Rotation', cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow('Original', cv2.WINDOW_AUTOSIZE)
        cv2.resizeWindow('Rotation', 1000, 1000)
        theta=float(self.angletext.Value)
        tx = self.txtext.Value
        ty = self.tytext.Value
        scale = float(self.scaletext.Value)
        a = cv2.imread('pyramid.jpg', 0)
        H=np.float32([[1, 0, tx],
                          [0, 1, ty]])
        #H = np.float32([[math.cos(theta), (-1)*math.sin(theta), tx],
                        #  [math.sin(theta), math.cos(theta), ty]])
        rows, cols = a.shape[:2]
        res = cv2.warpAffine(a, H, (rows, cols))
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), theta, 1)
        res1 = cv2.warpAffine(res, M, (rows, cols))
        res2 = cv2.resize(res1, (rows, cols), None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        cv2.imshow('Original', a)
        cv2.imshow('Rotation', res2)

    def OnClick52(self,event):
        a = cv2.imread('perspective.png', 0)
        cv2.namedWindow('Original Image')
        cv2.imshow('Original Image', a)
        pts2 = np.float32([[0, 0], [0, 300], [300, 300], [300, 0]])

        pos=[]
        def k(event, aax, aay, flags, param):
            global pts1
            if event == cv2.EVENT_LBUTTONDBLCLK:
                cv2.circle(a, (aax, aay), 30, (255, 255, 0), 0)
                p=[aax,aay]
                pos.append(p)
                print(pos)
                print(len(pos))
                pts1 = np.float32(pos)

        cv2.setMouseCallback('Original Image', k)
        while(1):
            if len(pos) != len(pts2):
                cv2.waitKey(1)
                pass
            if len(pos) == len(pts2):
                print(pts1)
                M = cv2.getPerspectiveTransform(pts1, pts2)
                print(M)
                dst = cv2.warpPerspective(a, M, (300, 300))
                cv2.namedWindow('Perspective Image')
                cv2.imshow('Perspective Image', dst)
                break

    def OnClick53(self,event):
        print(pts1)
        a = cv2.imread('perspective.png', 0)
        pts2 = np.float32([[0, 0], [0, 300] ,  [300, 300], [300, 0]])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        print(M)
        dst = cv2.warpPerspective(a, M, (300, 300))
        cv2.namedWindow('Perspective Image')
        cv2.imshow('Perspective Image', dst)

    def OnClick(self,  event):
        self.button11.SetLabel("Clicked")
        pass

def main():
    app = wx.App()
    WindowClass(None, 'title')
    app.MainLoop()

main()