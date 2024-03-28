import pygame
from main import guess

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

ORANGE = (255,165,0)
WHITE = (255,255,255)
BLACK = (0,0,0)

WIDTH = 500
HEIGHT = 400

BUTTON_HEIGHT = 20
BUTTON_WIDTH = 80
BUTTON_GAP = 50


PIXEL_SIZE = 10

PADDING_LEFT = 40
PADDING_TOP = 40

COMICSANS = pygame.font.SysFont("comicsans", 16)

WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Piirtoalusta")

state = [0]*28*28

def draw_canvas():
    global state
    WIN.fill(ORANGE)
    for i in range(len(state)):
        color = WHITE
        if state[i] == 1:
            color = BLACK
        WIN.fill(color, (i%28*PIXEL_SIZE+PADDING_LEFT, i//28*PIXEL_SIZE+PADDING_TOP, PIXEL_SIZE, PIXEL_SIZE))
    for button in Button.buttons:
        button.draw()
    pygame.display.update()


def set_state(newState):
    global state
    state = newState

def clear():
    set_state(784*[0])

def recognize():
    guess(state)

def main():
    run = True
    mousedown = False
    Button(PADDING_LEFT + 14*PIXEL_SIZE-BUTTON_GAP/2-BUTTON_WIDTH, PADDING_TOP+28*PIXEL_SIZE+(HEIGHT-(PADDING_TOP+28*PIXEL_SIZE))/2-BUTTON_HEIGHT, "Tyhjennä", clear)
    Button(PADDING_LEFT + 14*PIXEL_SIZE+BUTTON_GAP/2, PADDING_TOP+28*PIXEL_SIZE+(HEIGHT-(PADDING_TOP+28*PIXEL_SIZE))/2-BUTTON_HEIGHT, "Tunnista", recognize)
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


main()
