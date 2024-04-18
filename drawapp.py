import pygame
from done_nns.main_softmax import guess, learn, guessRandom
import numpy as np
from scipy.signal import convolve2d

learn()

pygame.font.init()


class Button():
    buttons = []
    def __init__(self, x, y, text, action) -> None:
        self.rect = pygame.Rect(x,y,BUTTON_WIDTH,BUTTON_HEIGHT)
        self.text = text
        self.action = action
        Button.buttons.append(self)
    def draw(self):
        WIN.fill(WHITE,self.rect)
        text = COMICSANS.render(self.text,1, BLACK)
        WIN.blit(text, (self.rect.x+BUTTON_WIDTH/2-text.get_width()/2,self.rect.y+BUTTON_HEIGHT/2-text.get_height()/2))
    def clicked(self):
        [x,y] = pygame.mouse.get_pos()
        x -= self.rect.x
        y -= self.rect.y
        return (0<x<BUTTON_WIDTH and 0<y<BUTTON_HEIGHT)
    def setText(self,txt):
        self.text = txt

button = None

ORANGE = (255,165,0)
WHITE = (255,255,255)
BLACK = (0,0,0)

WIDTH = 500
HEIGHT = 400

BUTTON_HEIGHT = 20
BUTTON_WIDTH = 100
BUTTON_GAP = 20
BUTTON_HOP = 16

RESULT_PADDING_TOP = 60
RESULT_GAP = 10

PIXEL_SIZE = 10

PADDING_LEFT = 40
PADDING_TOP = 40

COMICSANS = pygame.font.SysFont("comicsans", 16)
COMICSANS2 = pygame.font.SysFont("comicsans", 10)

WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Piirtoalusta")

state = np.zeros(784)
results = np.zeros(10)
showresults = False

def draw_canvas():
    global state
    WIN.fill(ORANGE)
    for i in range(len(state)):
        color = (((1-state[i])*255,(1-state[i])*255,(1-state[i])*255))
        WIN.fill(color, (i%28*PIXEL_SIZE+PADDING_LEFT, i//28*PIXEL_SIZE+PADDING_TOP, PIXEL_SIZE, PIXEL_SIZE))
    for button in Button.buttons:
        button.draw()
    result = COMICSANS.render(f"Arvaus: {np.argmax(results)} ({round(results.max()*100,1)}%)",1, BLACK)
    WIN.blit(result, (PADDING_LEFT + 28*PIXEL_SIZE+(WIDTH-(PADDING_LEFT + 28*PIXEL_SIZE))/2-result.get_width()/2, RESULT_PADDING_TOP))
    if showresults:
        for i in range(10):
            txt = COMICSANS2.render(f"{i}: {round(results[i]*100,1)}%",1, BLACK)
            WIN.blit(txt, (PADDING_LEFT + 28*PIXEL_SIZE+(WIDTH-(PADDING_LEFT + 28*PIXEL_SIZE))/2-txt.get_width()/2,RESULT_PADDING_TOP+txt.get_height()*(i+1)+BUTTON_HEIGHT+RESULT_GAP*(i+2)))
    pygame.display.update()


def set_state(newState):
    global state
    state = newState

def set_results(newResults):
    global results
    results = newResults

def clear():
    set_state(np.zeros(784))

def recognize():
    set_results(guess(state))

def getRandomGuess():
    state, predictions = guessRandom()
    set_state(state)
    set_results(predictions)

def blur():
    global state
    state = convolve2d(state.reshape((28,28)), np.ones((3,3))/9, mode="same").reshape((784))
def toggleShow():
    global showresults
    showresults = not showresults
    txt = "Näytä lisää"
    if showresults:
        txt = "Piilota"
    button.setText(txt)

def main():
    global button
    run = True
    mousedown = False
    Button(PADDING_LEFT + 14*PIXEL_SIZE-BUTTON_GAP-BUTTON_WIDTH*3/2, PADDING_TOP+28*PIXEL_SIZE+(HEIGHT-(PADDING_TOP+28*PIXEL_SIZE))/2-BUTTON_HEIGHT, "Tyhjennä", clear)
    Button(PADDING_LEFT + 14*PIXEL_SIZE-BUTTON_WIDTH/2, PADDING_TOP+28*PIXEL_SIZE+(HEIGHT-(PADDING_TOP+28*PIXEL_SIZE))/2-BUTTON_HEIGHT, "Sumenna", blur)
    Button(PADDING_LEFT + 14*PIXEL_SIZE+BUTTON_WIDTH/2+BUTTON_GAP, PADDING_TOP+28*PIXEL_SIZE+(HEIGHT-(PADDING_TOP+28*PIXEL_SIZE))/2-BUTTON_HEIGHT, "Tunnista", recognize)
    button = Button(PADDING_LEFT + 28*PIXEL_SIZE+(WIDTH-(PADDING_LEFT + 28*PIXEL_SIZE))/2-BUTTON_WIDTH/2, RESULT_PADDING_TOP+BUTTON_HOP+RESULT_GAP, "Näytä lisää", toggleShow)
    Button(PADDING_LEFT + 28*PIXEL_SIZE+(WIDTH-(PADDING_LEFT + 28*PIXEL_SIZE))/2-BUTTON_WIDTH/2, PADDING_TOP/2, "Satunnainen", getRandomGuess)
    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mousedown = True
                for button in Button.buttons:
                    if button.clicked():
                        button.action()
            elif event.type == pygame.MOUSEMOTION:
                [x, y] = pygame.mouse.get_pos()
                x -= PADDING_LEFT
                y -= PADDING_TOP
       
                if mousedown and 0<x<28*PIXEL_SIZE and 0<y<28*PIXEL_SIZE:
                    state[y//PIXEL_SIZE*28+x//PIXEL_SIZE] = 1
                    
            elif event.type == pygame.MOUSEBUTTONUP:
                mousedown = False
            
        draw_canvas()

if __name__ == "__main__":
    main()
