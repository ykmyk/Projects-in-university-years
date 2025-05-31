## Contorolling Arduino Uno with C ++
This is the final assignemnt(project) of Computer System course, which I took summer semester of 1st year.
This is the implementation to control arduino uno.

## Task specification
The objective is to create a random number generator that will simulate dice throws in the Advanced Dungeons&Dragons game. The game uses various types of polyhedral dice with different numbers of sides (d4, d6, d8, ...). Furthermore, a generated random number represents a sum of multiple throws. E.g., 3d6 means that player should throw 3 times using a dice with 6 sides (cube) and take the sum of these three throws (which is a number between 3 and 18).

The dice application is controlled by 3 buttons and displays the results on the 4-digit LED display. It operates in two modes. In normal mode, it displays the last generated number on the LED display. In configuration mode, it displays the type (and repetition factor) of dice being simulated. The first digit (1-9) displays the number of throws, the second digit displays the symbol 'd', and the remaining two digits indicate the type of dice (d4, d6, d8, d10, d12, d20, and d100 should be supported; d100 is displayed as '00' on the display).

### Button 1
- switches the dice to normal mode
- whilst pressed down, the random data are gathered (result is being generated)
- when the button is released, a new random result has to be displayed

### Button 2
- switches the dice to configuration mode
- increments the number of throws (if 9 is exceeded, the number returns to 1)

### Button 3
- switches the dice to configuration mode
- changes the dice type (dices d4-d100 from the list above are cycled)
- It might be a good idea to show some 'activity' on the display whilst the random number is being generated (i.e., when button 1 is pressed). You may show currently computed random numbers (if they change fast enough so the user cannot possibly stop at the right number), or you may create some sort of animation on the LED display or by the means of the other onboard LEDs.

Remember that the probability distribution is not uniform (for more than one die). Your simulator must use a counter or a time measurement of how long button 1 has been pressed to get a random number (which follows uniform distribution). You may use additional pseudo-random generators (Arduino built-in functions are adequate) to assist you with this task, but the initial randomness has to be tied somehow to the button event duration.

The details that are not specified here can be varied based on your creativity. Similarly, you may extend this assignment to your liking if such extensions do not reduce the amount of work. However, every decision you make should have a sound reason (i.e., you should be able to explain your decisions). If in doubt, consult your lab teacher.
