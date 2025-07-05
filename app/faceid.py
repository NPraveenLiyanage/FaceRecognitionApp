#Import kivy dependencies
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.logger import Logger

import cv2
import tensorflow as tf
from layers import L1Dist
import os 
import numpy as np

#Build app and layout
class CamApp(App):

    def build(self):
        #Main layout component
        self.web_cam = Image(size_hint = (1,.8))
        self.button = Button(text='Verify', size_hint=(1,.1))
        self.verification = Label(text='Verification Uninitiated', size_hint=(1,.1))

        #add items to layout
        layout = BoxLayout(orientation= 'vertical')
        layout.add_widget(self.web_cam)
        layout.add_widget(self.verification)
        layout.add_widget(self.button)

        #setup video capture device
        self.capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.update, 1.0/33.0)

        return layout

    #Run continuously to get web cam feed
    def update(self, *args):

        #Read frame from openCv
        ret, frame = self.capture.read()

        #cut down frame to 250x250px
        frame = frame[120:120+250,200:200+250,:] 

        #Flip horizontall and convert image to texture
        buf = cv2.flip(frame, 0).tostring()
        img_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt = 'bgr')
        img_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.web_cam.texture = img_texture

if __name__ == '__main__':
    CamApp().run() 