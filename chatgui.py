from tkinter import *
from datetime import datetime
import random
import re
from tkinter import messagebox
from tkinter.font import Font
import textwrap
from main import chatbot_response

root = Tk()
root.config(bg="lightblue")
root.geometry('410x600+400+100')

canvas = Canvas(root, width=200, height=200,bg="white")
canvas.grid(row=0,column=0,columnspan=2)
canvas.place(x=10, y=10, width=390, height=530)

texts = []

class TextBot:
    def __init__(self,master,message=""):
        self.master = master
        self.frame = Frame(master,bg="light green")
        self.i = self.master.create_window(10,450,window=self.frame, anchor="w")
        Label(self.frame,text="You", font=("Helvetica", 11),bg="light green").grid(row=0,column=1,sticky="w",padx=5)
        Label(self.frame,text= datetime.now().strftime("%d-%m-%Y %X"),font=("Helvetica", 9),bg="light green").grid(row=1,column=1,sticky="w",padx=5)
        Label(self.frame, text=textwrap.fill(message, 25), font=("Helvetica", 13),bg="light green").grid(row=2, column=1,sticky="w",padx=5,pady=3)
        root.update_idletasks()

class ReplyBot:
    def __init__(self,master,message=""):
        self.master = master
        self.frame = Frame(master,bg="light blue")
        self.i = self.master.create_window(10,400,window=self.frame, anchor="w")
        Label(self.frame,text="eTA", font=("Helvetica", 11),bg="light blue").grid(row=0,column=1,sticky="w",padx=5)
        Label(self.frame,text= datetime.now().strftime("%d-%m-%Y %X"),font=("Helvetica", 9),bg="light blue").grid(row=1,column=1,sticky="w",padx=5)
        Label(self.frame, text=textwrap.fill(message, 25), font=("Helvetica", 13),bg="light blue").grid(row=2, column=1,sticky="w",padx=5,pady=3)
        root.update_idletasks()

def send_message():
    if texts:
        x = len(texts[0]) // 25 * 30 + 100
        canvas.move(ALL, 0, -x)
    TextBot(canvas,message=ChatLog.get())
    msg = ChatLog.get()
    texts.append(msg)
    y = len(msg) // 25 * 50 + 150
    canvas.move(ALL, 0, -y)
    res = chatbot_response(msg)
    ReplyBot(canvas, message=res)
    texts.append(res)
    ChatLog.delete(0,'end')


ChatLog = Entry(root,width=26, font=("Helvetica", 13))
ChatLog.place(x=10, y=550, width=290, height=40)


#buton
SendButton = Button(root, width=10, height=2,
relief='raised',state='active',command=send_message)
SendButton.config(text='Send', bg='lightblue', font='Verdana 13')
SendButton.place(x=310, y=550)

root.mainloop()