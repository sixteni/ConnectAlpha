import numpy as np
EMPTY = 0
class ConnectXRules():
    COLLS = 7
    ROWS = 6


    @staticmethod
    def is_terminal_state(state, coll, color, move_count, has_played = True):        
        '''Checks if a move results in ending the game e.i. win/draw

        Args:
            state (np.array): The current state
            pos (int): The position in where the last move was made, pos < ROWS * COLLS
            color (int): the color of the stone 1 = black, -1=white
            move_count (int): the move count 1,2,3....ROWS * COLLS
            has_played(bool): Has already played the move or not
        Returns:
            tuple(bool, int): is game ending move, the value of the game (1=current player wins, 0=draw, None=game continous)
        '''
                
        if has_played:
          for j in range(ConnectXRules.ROWS):
            if state[j][coll] != EMPTY:
              y = j
              break  
        else:
          for j in range(ConnectXRules.ROWS):
            if state[ConnectXRules.ROWS - 1 - j][coll] == EMPTY:
              y = ConnectXRules.ROWS - 1 - j
              break      
 
        x = coll
        
        if ConnectXRules.is_winning_move(state, y, x, color):
          return True, 1
        
        if move_count == ConnectXRules.COLLS * ConnectXRules.ROWS:
          return True, 0

        return False, None

    @staticmethod
    def make_move(state, coll, color):
        '''Makes a move in the current state

        Args:
            state (np.array): The current state
            pos (int): The position in whitch to make the move, pos < ROWS * COLLS
            color (int): the color of the stone 1 = black, -1=white

        Returns:
            np.array: The new state after the move
        '''
        new_state = np.array(state)
        for y in range(ConnectXRules.ROWS):
          if new_state[ConnectXRules.ROWS - 1 - y][coll] == EMPTY:
            new_state[ConnectXRules.ROWS - 1 - y][coll] = color
            break       
        return new_state

    @staticmethod
    def get_legal_actions(state):
        '''Gets all valid moves in the current state

        Args:
            state (np.array): The current state

        Returns:
            list: One dimensional vector containing all valid position indexes
        '''

        valid_moves = []
        for x in range(ConnectXRules.COLLS):
            if ConnectXRules.is_valid_move(state, 0, x):
              valid_moves.append(x)
        return valid_moves

    @staticmethod
    def is_valid_move(state, y, x):
        '''Checks if a move is valid

        Args:
            state (np.array): The current state
            y (int): The row position
            x (int): The collumn position

        Returns: 
          bool: Is valid or not
        '''

        return ConnectXRules.is_in_bounds( y, x) and state[y][x] == EMPTY 

    @staticmethod
    def is_winning_move(state, y, x, color):
        '''Checks if a move is results in a win for the current player

        Args:
            state (np.array): The current state
            y (int): The row position
            x (int): The collumn position

        Returns: 
          bool: Is winning move
        '''
     
        winning_move = False

        # Directions: Horizontal, Vertical, SW->NE, NW->SE        
        directions = [(0,1), (1,0), (1,-1), (-1,-1)]

        for dy, dx in directions:
          winning_move = ConnectXRules.is_winnig_line(state, y, x, dy, dx, color)
          if winning_move: 
            break
        
        return winning_move

    @staticmethod
    def is_winnig_line(state, y, x, dy, dx, color):
        '''Checks how many disks of the same color are connected in a line

        Args:
            state (np.array): The current state
            y (int): The center y position in the line
            x (int): The center x position in the line
            dy (int): The step to take in y direction
            dx (int): The step to take in x direction
            color (int): The color of the current player

        Returns: 
          bool: Is winning line (i.e. 4 consecutive stones)
        '''
        in_a_row = ConnectXRules.check_adjacent_side(state, y, x, dy, dx, color) + 1
        if(in_a_row >= 4):
          return True
        in_a_row += ConnectXRules.check_adjacent_side(state, y, x, -dy, -dx, color)
        return in_a_row >= 4

    @staticmethod
    def check_adjacent_side(state, y, x, dy, dx, color):
        '''
            Checks the how many disks are connected to the current move
            on one adjacent side (right/left, up/down, SE/NW, SW/NE )

            returns: connected_count
        '''

        connected_count = 0
        
        # Moves maximum 3 steps from the current move pos
        for i in range(3):
          y2 = y + dy * (i + 1)
          x2 = x + dx * (i + 1)
          if ConnectXRules.is_in_bounds(y2, x2) and state[y2][x2] == color:
            connected_count += 1            
          else:
            break

        return connected_count

    @staticmethod
    def is_in_bounds(y, x):
        '''Checks if a move is inside the board

        Args:            
            y (int): The row position
            x (int): The collumn position

        Returns: 
          bool: Is inside the game board
        '''
        return x < ConnectXRules.COLLS and x >= 0 and y < ConnectXRules.ROWS and y >= 0