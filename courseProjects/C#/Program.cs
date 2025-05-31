using System;
using System.Collections.Generic;
using static System.Console;

public class GameBoard
{
    private char[,] board;
    public const int Size = 6;  // Updated board size to 6x6

    public GameBoard()
    {
        board = new char[Size, Size];
        InitializeBoard();
    }

    private void InitializeBoard()
    {
        for (int i = 0; i < Size; i++)
            for (int j = 0; j < Size; j++)
                board[i, j] = ' ';
    }

    public char[,] BoardState
    {
        get { return board; }
    }

    public bool IsValidMove(int x, int y)
    {
        if (IsEmpty())
            return true;
        if (board[x, y] != ' ') return false;
        return HasNeighbor(x, y);
    }

    private bool HasNeighbor(int x, int y)
    {
        for (int i = -1; i <= 1; i++)
        {
            for (int j = -1; j <= 1; j++)
            {
                if (i == 0 && j == 0) continue;
                int nx = x + i, ny = y + j;
                if (nx >= 0 && ny >= 0 && nx < Size && ny < Size && board[nx, ny] != ' ')
                    return true;
            }
        }
        return false;
    }

    public bool MakeMove(int x, int y, char player)
    {
        if (!IsValidMove(x, y)) return false;
        board[x, y] = player;
        return true;
    }

    public void UndoMove(int x, int y)
    {
        board[x, y] = ' ';
    }

    public bool CheckWin(char player)
    {
        // Check horizontal lines
        for (int i = 0; i < Size; i++)
        {
            for (int j = 0; j < Size - 3; j++)
            {
                if (board[i, j] == player && board[i, j + 1] == player && board[i, j + 2] == player && board[i, j + 3] == player)
                {
                    return true;
                }
            }
        }

        // Check vertical lines
        for (int i = 0; i < Size - 3; i++)
        {
            for (int j = 0; j < Size; j++)
            {
                if (board[i, j] == player && board[i + 1, j] == player && board[i + 2, j] == player && board[i + 3, j] == player)
                {
                    return true;
                }
            }
        }

        // Check diagonal lines (top-left to bottom-right)
        for (int i = 0; i < Size - 3; i++)
        {
            for (int j = 0; j < Size - 3; j++)
            {
                if (board[i, j] == player && board[i + 1, j + 1] == player && board[i + 2, j + 2] == player && board[i + 3, j + 3] == player)
                {
                    return true;
                }
            }
        }

        // Check diagonal lines (top-right to bottom-left)
        for (int i = 0; i < Size - 3; i++)
        {
            for (int j = 3; j < Size; j++)
            {
                if (board[i, j] == player && board[i + 1, j - 1] == player && board[i + 2, j - 2] == player && board[i + 3, j - 3] == player)
                {
                    return true;
                }
            }
        }

        return false;
    }

    public bool IsFull()
    {
        for (int i = 0; i < Size; i++)
        {
            for (int j = 0; j < Size; j++)
            {
                if (board[i, j] == ' ')
                {
                    return false;
                }
            }
        }
        return true;
    }

    public bool IsEmpty()
    {
        for (int i = 0; i < Size; i++)
        {
            for (int j = 0; j < Size; j++)
            {
                if (board[i, j] != ' ')
                {
                    return false;
                }
            }
        }
        return true;
    }

    public List<(int x, int y)> GetAvailableMoves()
    {
        List<(int x, int y)> moves = new List<(int x, int y)>();
        for (int i = 0; i < Size; i++)
        {
            for (int j = 0; j < Size; j++)
            {
                if (IsValidMove(i, j))
                {
                    moves.Add((i, j));
                }
            }
        }
        return moves;
    }

    public void PrintBoard()
    {
        // Print column numbers
        Write(" ");
        for (int col = 0; col < Size; col++)
        {
            Write($"  {col} ");
        }
        WriteLine();

        for (int i = 0; i < Size; i++)
        {
            // Print row number
            Write($"{i}  ");

            for (int j = 0; j < Size; j++)
            {
                Write(board[i, j] == ' ' ? " " : board[i, j].ToString());
                if (j < Size - 1)
                {
                    Write(" | ");
                }
            }
            WriteLine();

            // Print row separator, but not after the last row
            if (i < Size - 1)
            {
                Write("  ");
                WriteLine(new string('-', Size * 4 - 1));
            }
        }
    }

}

public interface IPlayer
{
    (int x, int y) GetMove(GameBoard board); // Interface to retrieve the next move
    char Symbol { get; } // Every player has a symbol (either 'X' or 'O')
}

public abstract class Player
{
    public char Symbol { get; }

    protected Player(char symbol)
    {
        Symbol = symbol;
    }

    public abstract (int x, int y) GetMove(GameBoard board);
}
public class HumanPlayer : IPlayer
{
    public char Symbol { get; }

    public HumanPlayer(char symbol)
    {
        Symbol = symbol;
    }

    public (int x, int y) GetMove(GameBoard board)
    {
        Console.WriteLine("Enter your move (row and column): ");
        int x = int.Parse(Console.ReadLine());
        int y = int.Parse(Console.ReadLine());
        return (x, y);
    }
}

public class AIPlayer : IPlayer
{
    public char Symbol { get; }
    private int maxDepth;

    public AIPlayer(char symbol, int maxDepth = 3)
    {
        Symbol = symbol;
        this.maxDepth = maxDepth;
    }

    public (int x, int y) GetMove(GameBoard board)
    {
        (int x, int y) bestMove = (-1, -1);
        int bestValue = int.MinValue;

        foreach (var move in board.GetAvailableMoves())
        {
            board.MakeMove(move.x, move.y, Symbol);
            int moveValue = Minimax(board, 0, false, int.MinValue, int.MaxValue);
            board.UndoMove(move.x, move.y);

            if (moveValue > bestValue)
            {
                bestMove = (move.x, move.y);
                bestValue = moveValue;
            }
        }
        return bestMove;
    }

    private int Minimax(GameBoard board, int depth, bool isMaximizing, int alpha, int beta)
    {
        char opponentSymbol = (Symbol == 'X') ? 'O' : 'X';

        if (depth >= maxDepth || board.CheckWin('X') || board.CheckWin('O') || board.IsFull())
        {
            return EvaluateBoard(board);
        }

        if (isMaximizing)
        {
            int maxEval = int.MinValue;
            foreach (var move in board.GetAvailableMoves())
            {
                board.MakeMove(move.x, move.y, Symbol);
                int eval = Minimax(board, depth + 1, false, alpha, beta);
                board.UndoMove(move.x, move.y);
                maxEval = Math.Max(maxEval, eval);
                alpha = Math.Max(alpha, eval);
                if (beta <= alpha)
                    break;
            }
            return maxEval;
        }
        else
        {
            int minEval = int.MaxValue;
            foreach (var move in board.GetAvailableMoves())
            {
                board.MakeMove(move.x, move.y, opponentSymbol);
                int eval = Minimax(board, depth + 1, true, alpha, beta);

                if (IsCriticalMove(board, move.x, move.y, opponentSymbol))
                {
                    eval -= 10000; // Discourage moves that allow the opponent to win
                }

                board.UndoMove(move.x, move.y);
                minEval = Math.Min(minEval, eval);
                beta = Math.Min(beta, eval);
                if (beta <= alpha)
                    break;
            }
            return minEval;
        }
    }
    private bool IsCriticalMove(GameBoard board, int x, int y, char opponentSymbol)
    {
        // Check all four directions for a critical line
        return CheckPotentialLine(board, x, y, 0, 1, opponentSymbol) ||   // Horizontal
            CheckPotentialLine(board, x, y, 1, 0, opponentSymbol) ||   // Vertical
            CheckPotentialLine(board, x, y, 1, 1, opponentSymbol) ||   // Diagonal top-left to bottom-right
            CheckPotentialLine(board, x, y, 1, -1, opponentSymbol);    // Diagonal top-right to bottom-left
    }


    private bool CheckPotentialLine(GameBoard board, int startX, int startY, int dx, int dy, char opponentSymbol)
    {
        int opponentCount = 0;
        int emptyCount = 0;

        for (int i = 0; i < 4; i++)
        {
            int x = startX + i * dx;
            int y = startY + i * dy;
            if (x >= 0 && x < GameBoard.Size && y >= 0 && y < GameBoard.Size)
            {
                if (board.BoardState[x, y] == opponentSymbol)
                {
                    opponentCount++;
                }
                else if (board.BoardState[x, y] == ' ')
                {
                    emptyCount++;
                }
            }
        }

        // Consider the line critical if the opponent has 3 stones and the rest are empty.
        return opponentCount == 3 && emptyCount == 1;
    }


    private int EvaluateBoard(GameBoard board)
    {
        int score = 0;

        // Evaluate rows (horizontal)
        for (int i = 0; i < GameBoard.Size; i++)
        {
            for (int j = 0; j <= GameBoard.Size - 4; j++) // Start positions for horizontal
            {
                score += EvaluateLine(board, i, j, 0, 1); // Horizontal line
            }
        }

        // Evaluate columns (vertical)
        for (int j = 0; j < GameBoard.Size; j++)
        {
            for (int i = 0; i <= GameBoard.Size - 4; i++) // Start positions for vertical
            {
                score += EvaluateLine(board, i, j, 1, 0); // Vertical line
            }
        }

        // Evaluate diagonals (top-left to bottom-right)
        for (int i = 0; i <= GameBoard.Size - 4; i++) // Start positions for top-left to bottom-right
        {
            for (int j = 0; j <= GameBoard.Size - 4; j++)
            {
                score += EvaluateLine(board, i, j, 1, 1); // Diagonal from top-left to bottom-right
            }
        }

        // Evaluate diagonals (top-right to bottom-left)
        for (int i = 0; i <= GameBoard.Size - 4; i++) // Start positions for top-right to bottom-left
        {
            for (int j = 3; j < GameBoard.Size; j++)
            {
                score += EvaluateLine(board, i, j, 1, -1); // Diagonal from top-right to bottom-left
            }
        }

        return score;
    }


    private int EvaluateLine(GameBoard board, int startX, int startY, int dx, int dy)
    {
        int aiCount = 0;
        int opponentCount = 0;
        int emptyCount = 0;
        char aiSymbol = Symbol;
        char opponentSymbol = (Symbol == 'X') ? 'O' : 'X';

        for (int i = 0; i < 4; i++) // Check for up to 4 in a row
        {
            int x = startX + i * dx;
            int y = startY + i * dy;
            if (x >= GameBoard.Size || y >= GameBoard.Size || x < 0 || y < 0)
                break;

            if (board.BoardState[x, y] == aiSymbol)
                aiCount++;
            else if (board.BoardState[x, y] == opponentSymbol)
                opponentCount++;
            else if (board.BoardState[x, y] == ' ')
                emptyCount++;
        }

        // Winning and blocking conditions
        if (aiCount == 4) return 100000; // AI win
        if (opponentCount == 4) return -100000; // Opponent win
        if (opponentCount == 3 && emptyCount == 1) return -10000; // Strongly block opponent's potential win
        if (aiCount == 3 && emptyCount == 1) return 10000; // Strongly favor creating a winning opportunity

        // Adjust score based on AI stones and opponent stones
        return aiCount * 10 - opponentCount * 10;
    }


}

public class Game
{
    private GameBoard board;
    private IPlayer player1;
    private IPlayer player2;
    private IPlayer currentPlayer;

    public Game(IPlayer p1, IPlayer p2)
    {
        board = new GameBoard();
        player1 = p1;
        player2 = p2;
        currentPlayer = player1;
    }

    public void Start()
{
    // Print the initial empty board before any move
    board.PrintBoard();

    while (true)
    {
        // Get the move from the current player
        (int x, int y) move = currentPlayer.GetMove(board);
        
        // Deconstruct the move tuple into x and y
        int x = move.x;
        int y = move.y;

        if (board.IsValidMove(x, y))
        {
            // Make the move and print the board
            board.MakeMove(x, y, currentPlayer.Symbol);
            board.PrintBoard();

            // Check if the current player has won
            if (board.CheckWin(currentPlayer.Symbol))
            {
                Console.WriteLine($"{currentPlayer.Symbol} wins!");
                break;
            }

            // Switch to the other player
            currentPlayer = (currentPlayer == player1) ? player2 : player1;
        }
        else
        {
            Console.WriteLine("Invalid move. Try again.");
        }
    }
}


    public static void Main(string[] args)
    {
        IPlayer human = new HumanPlayer('X');
        IPlayer ai = new AIPlayer('O', 3); // Adjust the depth limit here
        Game game = new Game(human, ai);
        game.Start();
    }
}
