import PySimpleGUI as sg
from PIL import ImageTk


# Function that creates a window and returns it.
# Makes use of the PySimpleGUI library.
def create_window():
    sg.theme('Reddit')

    layout = create_layout()
    window = sg.Window('BIAI', layout, size=(600, 500), element_justification='center')
    return window


# Function that creates a layout for the window.
# Returns a layout later used by create_window().
def create_layout():
    layout = [
        [sg.Text('Choose an image to predict:')],
        [sg.Input(key="-IN-", enable_events=True), sg.FileBrowse()],
        [sg.Image(size=(300, 300), key="-IMAGE-")],
        [sg.Button('Predict', size=(10, 2))],
        [sg.Text('Prediction:', key='-PREDICTION-', font=('Helvetica', 14), text_color='#333')],
    ]
    return layout


# Function used to update the image in the window after loading it from a file.
def update_image(window, image):
    image_converted = ImageTk.PhotoImage(image=image)
    window['-IMAGE-'].update(data=image_converted)


# Function used to update the prediction in the window after making a prediction.
def update_prediction(window, prediction, score):
    window['-PREDICTION-'].update("Prediction: " + prediction + " (" + str(round(score)) + "%)", text_color='#333')

