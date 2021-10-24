#Creating GUI with tkinter
import tkinter
from tkinter import *

from datetime import datetime
import random
import re
from tkinter import messagebox
from tkinter.font import Font
import textwrap



def send():
    msg = EntryBox.get("1.0",'end-1c').strip()
    EntryBox.delete("0.0",END)

    if msg != '':
        ChatLog.config(state=NORMAL)
        ChatLog.insert(END, "You: " + msg + '\n\n')
        ChatLog.config(foreground="#442265", font=("Verdana", 15 ))

        res = chatbot_response(msg)
        ChatLog.insert(END, "Bot: " + res + '\n\n')

        ChatLog.config(state=DISABLED)
        ChatLog.yview(END)

#background and frame
root = Tk()
root.config(bg="lightblue")
root.geometry('410x600+400+100')

#chat window
ChatLog = Text(root, width=200, height=200,bg="white")
ChatLog.grid(row=0,column=0,columnspan=2)
ChatLog.place(x=10, y=10, width=390, height=530)

scrollbar = Scrollbar(root, command=ChatLog.yview, cursor="heart")
ChatLog['yscrollcommand'] = scrollbar.set

SendButton = Button(root, width=10, height=2,text = "Send", command = send)

SendButton.place(x=310, y=550)

EntryBox = Text(root,width=26, font=("Verdana", 15))
EntryBox.place(x=10, y=550, width=290, height=40)


root.mainloop()
