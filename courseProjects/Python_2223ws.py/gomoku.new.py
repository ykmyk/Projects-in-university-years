import random

# numbers of line in the board
Lines = 9

# players
You = 0
Me = 1

# color of stones
your_stone = 'black'
my_stone = 'white'

class Gomoku:

    def __init__(self):
        self.board = None    

    def player_selection(self):
        self.board = [[None] * Lines for i in range(Lines)]
        print("Do you want to start? \nenter which to start, 'me' or 'computer'")
        starter = input()
        if starter == 'me' or starter == "'me'":
            self.turns(2)
        elif starter == 'computer' or starter == "'computer'":
            self.turns(3)
        else:
            print("please enter corrct player's name to start. \n")
            self.player_selection()

    def turns(self, player):
        if player == 0:
            x = int(input("select the number for 'x': ")) - 1
            y = int(input("select the number for 'y' : ")) - 1
            if x < 0 or x >= 9 or y < 0 or y >= 9:
                print('please select x and y from 1 to 9')
                self.turns(0)
            elif self.board[y][x] != None:
                print('select unplaced position')
                self.turns(0)
            elif self.neighbor_checker(x, y) == False:
                print("you can't chose the place where there is any neighbor")
                self.turns(0)
            else:
                self.board[y][x] = your_stone
                self.table_print(self.board)

            if self.counts(x, y, your_stone) >= 5: 
                print('result', x, y)
                self.result(0)
                return
            self.turns(1)

        elif player == 1:
            mx, my = self.decition()
            if self.counts(mx, my, my_stone) >= 5:
                self.result(1)
                return
            self.table_print(self.board)
            self.turns(0)

        elif player == 2: # if the game is started by player
            print('please select x and y from the below and enter the number.', end='\n')
            self.table_print(self.board)
            x = int(input("select the number for 'x': ")) - 1
            y = int(input("select the number for  'y' : ")) - 1
            if x < 0 or x >= 9 or y < 0 or y >= 9:
                print('please select x and y from 1 to 9')
                self.turns(2)
            elif self.board[y][x] != None:
                print('select unplaced position')
                self.turns(2)
            else:
                self.board[y][x] = your_stone
                self.table_print(self.board)

            self.turns(1)

        else: # if player == 3 => if the game is started by computer(this is used in only first turn)
            self.first_decition()
            self.table_print(self.board)
            self.turns(0)

    def neighbor_checker(self, x, y):
        dir = [
        (1, 0), #right
        (1, 1), #right-down
        (0, 1), #down
        (-1, 1), #left-down
        (-1, 0), #left
        (-1, -1), #left-up
        (0, -1), # up
        (1, -1) ] # righ-up
        for i, j in dir:
            x1 = x + i
            y1 = y + j
            if x1 < 0 or x1 >= Lines or y1 < 0 or y1 >= Lines:
                continue
            if self.board[y1][x1] != None:
                return True

        return False

    def counts(self, x, y, stone):
        dir = [
        (1, 0), 
        (1, 1), 
        (0, 1), 
        (-1, 1) 
         ] 
        max_num = 0

        for i, j in dir:
            count_num = 1
            for k in range(1, Lines):
                x1 = x + i * k
                y1 = y + j * k
                if x1 < 0 or x1 >= Lines or y1 < 0 or y1 >= Lines:
                    break
                elif self.board[y1][x1] != stone or self.board[y1][x1] == None:
                    break
                else:
                    count_num += 1

            for k in range(-1, -(Lines), -1): 
                x2 = x + i * k
                y2 = y + j * k
                if x2 < 0 or x2 >= Lines or y2 < 0 or y2 >= Lines:
                    break
                elif self.board[y2][x2] != stone or self.board[y2][x2] == None:
                    break
                else:
                    count_num += 1

            if max_num < count_num:
                max_num = count_num

        return max_num

    def decition(self): # computer decides where to put the stones
        player_max_list = []
        max_ = 0
        for x in range(Lines):
            for y in range(Lines):
                if self.board[y][x] == None: 
                    count_num = self.counts(x, y, your_stone)

                    if count_num == max_:
                        player_max_list.append((x, y))

                    elif count_num > max_:
                        player_max_list = []
                        player_max_list.append((x, y))
                        max_ = count_num

        x, y = self.select_position(player_max_list)
        self.board[y][x] = my_stone
        return x, y
    
    def first_decition(self):
        x, y = 4, 4
        self.board[y][x] = my_stone
        return

    def select_position(self, max_list):
        select = random.randrange(len(max_list))
        x, y = max_list[select]
        if self.neighbor_checker(x, y) == True:
            return x, y
        else:
            del max_list[select]
            self.select_position(max_list)


    def result(self, winner): 
        if winner == 0:
            print('Result: YOU WIN!!!!')

        else:
            print('Result: YOU LOSE.....')


    def table_print(self, list):
        print('your stone: ○ ')
        print("computer's stone: ● ")
        print("y↓  x→ 1   2   3   4   5   6   7   8   9   ")
        print("     +---+---+---+---+---+---+---+---+---+ ")
        for i in range(0, 9):
            print(i + 1, "   |", end="")
            for j in range(0, 9):
                if list[i][j] == 'black':
                    print(" ○ ", end="|")
                elif list[i][j] == 'white':
                    print(" ● ", end="|")
                else:
                    print("   ", end="|")
            print('\n     +---+---+---+---+---+---+---+---+---+ ')



g = Gomoku()
g.player_selection()