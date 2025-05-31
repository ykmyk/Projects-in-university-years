#include "funshield.h"

// LED Display Class
class LEDDisplay {
public:
    LEDDisplay() {
        cleanDisplay();
    }

    void cleanDisplay() {
        for (int i = 0; i < 4; ++i) {
            LED[i] = emptyGlyph;
        }
    }

    void displayNumber(int number) {
        int len = numLen(number);
        int numOfTen = pow(10, len - 1);
        int numOf0 = 4 - len;
        for (int i = 0; i < numOf0; i++) {
            LED2[i] = 0;
        }
        for (int i = numOf0; i < 4; i++) {
            LED2[i] = (number / numOfTen) % 10;
            numOfTen /= 10;
        }
        updateDisplay();
    }

    void displayConfig(int throws, int diceType) {
        cleanDisplay();
        LED[0] = digits[throws];
        LED[1] = letterGlyph[3]; // 'd'
        if (diceType < 10) {
            LED[2] = digits[diceType];
            LED[3] = emptyGlyph;
        } else {
            displayDiceType(diceType);
        }
        updateDisplay();
    }

    void showAnimation() {
        for (int i = 0; i < 4; i++) {
            size_t rd = random(0, 10);
            LED[i] = digits[rd];
        }
        updateDisplay();
    }

    void updateDisplay() {
        for (int i = 0; i < 4; ++i) {
            displayDigit(LED[i], digit_muxpos[i]);
        }
    }

private:
    byte LED[4];
    int LED2[4];

    void displayDigit(byte digit, byte pos) {
        shiftOut(data_pin, clock_pin, MSBFIRST, digit);
        shiftOut(data_pin, clock_pin, MSBFIRST, pos);
        digitalWrite(latch_pin, LOW);
        digitalWrite(latch_pin, HIGH);
    }

    int numLen(int num) {
        return num > 0 ? (int)log10((double)num) + 1 : 1;
    }

    void displayDiceType(int diceType) {
        if (diceType == 100) {
            LED[2] = digits[0];
            LED[3] = digits[0];
        } else {
            LED[2] = digits[diceType / 10];
            LED[3] = digits[diceType % 10];
        }
    }
};

// Button Handler Class
class ButtonHandler {
public:
    ButtonHandler(int pin1, int pin2, int pin3)
        : buttonPins{ pin1, pin2, pin3 } {
        for (int i = 0; i < 3; ++i) {
            pinMode(buttonPins[i], INPUT);
            buttonStates[i] = OFF;
            previousState[i] = OFF;
        }
    }

    void readButtons() {
        for (int i = 0; i < 3; ++i) {
            buttonStates[i] = digitalRead(buttonPins[i]);
        }
    }

    int getButtonState(int button) const {
        return buttonStates[button];
    }

    bool isButtonPressed(int button) const {
        return buttonStates[button] == ON && previousState[button] == OFF;
    }

    bool isButtonReleased(int button) const {
        return buttonStates[button] == OFF && previousState[button] == ON;
    }

    void updatePreviousStates() {
        for (int i = 0; i < 3; ++i) {
            previousState[i] = buttonStates[i];
        }
    }

private:
    const int buttonPins[3];
    int buttonStates[3];
    int previousState[3];
};

// Dice Simulator Class
class DiceSimulator {
public:
    DiceSimulator()
        : numOfThrows(1), indexOfTypeOfDice(0), result(0), isNormalMode(true) {}

    void increaseThrows() {
        numOfThrows = (numOfThrows == 9) ? 1 : numOfThrows + 1;
    }

    void cycleDiceType() {
        indexOfTypeOfDice = (indexOfTypeOfDice + 1) % 7;
    }

    void throwDice(unsigned long seed) {
        randomSeed(seed + random());
        result = 0;
        for (int i = 0; i < numOfThrows; ++i) {
            result += random(1, typesOfDice[indexOfTypeOfDice] + 1);
        }
    }

    int getResult() const {
        return result;
    }

    int getNumOfThrows() const {
        return numOfThrows;
    }

    int getDiceType() const {
        return typesOfDice[indexOfTypeOfDice];
    }

    bool isInNormalMode() const {
        return isNormalMode;
    }

    void setNormalMode(bool mode) {
        isNormalMode = mode;
    }

private:
    int numOfThrows;
    int typesOfDice[7] = { 4, 6, 8, 10, 12, 20, 100 };
    int indexOfTypeOfDice;
    int result;
    bool isNormalMode;
};

// Main Arduino Sketch
LEDDisplay ledDisplay;
ButtonHandler buttonHandler(button1_pin, button2_pin, button3_pin);
DiceSimulator diceSimulator;

unsigned long activationDelayTime = 0;
constexpr unsigned long animRefreshTime = 70;
unsigned long prevAnimTime = 0;
unsigned long t, s;

void setup() {
    pinMode(latch_pin, OUTPUT);
    pinMode(clock_pin, OUTPUT);
    pinMode(data_pin, OUTPUT);
}

void loop() {
    t = micros();
    s = millis();
    buttonHandler.readButtons();

    if (buttonHandler.getButtonState(0) == ON) {
        ledDisplay.showAnimation();
    }

    if (buttonHandler.isButtonPressed(0)) {
        activationDelayTime = t;
    }

    if (buttonHandler.isButtonReleased(0)) {
        diceSimulator.throwDice(t - activationDelayTime);
        diceSimulator.setNormalMode(true);
        ledDisplay.displayNumber(diceSimulator.getResult());
    }

    if (buttonHandler.isButtonPressed(1)) {
        diceSimulator.increaseThrows();
        diceSimulator.setNormalMode(false);
    }

    if (buttonHandler.isButtonPressed(2)) {
        diceSimulator.cycleDiceType();
        diceSimulator.setNormalMode(false);
    }

    if (!diceSimulator.isInNormalMode()) {
        ledDisplay.displayConfig(diceSimulator.getNumOfThrows(), diceSimulator.getDiceType());
    }

    buttonHandler.updatePreviousStates();
}
