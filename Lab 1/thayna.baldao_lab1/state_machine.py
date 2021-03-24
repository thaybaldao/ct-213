import random
import math
from constants import *


class FiniteStateMachine(object):
    """
    A finite state machine.
    """

    def __init__(self, state):
        self.state = state

    def change_state(self, new_state):
        self.state = new_state

    def update(self, agent):
        self.state.check_transition(agent, self)
        self.state.execute(agent)


class State(object):
    """
    Abstract state class.
    """

    def __init__(self, state_name):
        """
        Creates a state.

        :param state_name: the name of the state.
        :type state_name: str
        """
        self.state_name = state_name

    def check_transition(self, agent, fsm):
        """
        Checks conditions and execute a state transition if needed.

        :param agent: the agent where this state is being executed on.
        :param fsm: finite state machine associated to this state.
        """
        raise NotImplementedError("This method is abstract and must be implemented in derived classes")

    def execute(self, agent):
        """
        Executes the state logic.

        :param agent: the agent where this state is being executed on.
        """
        raise NotImplementedError("This method is abstract and must be implemented in derived classes")


class MoveForwardState(State):
    def __init__(self):
        super().__init__("MoveForward")
        self.number_executions = 0  # counts the number of executions of this state

    def check_transition(self, agent, state_machine):
        # check if agent bumped
        if agent.get_bumper_state():
            state_machine.change_state(GoBackState())
        else:
            # compute current time
            dt = self.number_executions * SAMPLE_TIME
            # check if agent spent enough time going forward
            if dt > MOVE_FORWARD_TIME:
                state_machine.change_state(MoveInSpiralState())

    def execute(self, agent):
        self.number_executions += 1

        # make agent go forward
        agent.set_velocity(FORWARD_SPEED, 0)


class MoveInSpiralState(State):
    def __init__(self):
        super().__init__("MoveInSpiral")
        self.number_executions = 0  # counts the number of executions of this state

    def check_transition(self, agent, state_machine):
        # check if agent bumped
        if agent.get_bumper_state():
            state_machine.change_state(GoBackState())
        else:
            # compute current time
            dt = self.number_executions * SAMPLE_TIME
            # check if agent spent enough time moving in spiral
            if dt > MOVE_IN_SPIRAL_TIME:
                state_machine.change_state(MoveForwardState())

    def execute(self, agent):
        self.number_executions += 1

        # compute current time
        dt = self.number_executions * SAMPLE_TIME
        # compute current spiral radius
        radius = INITIAL_RADIUS_SPIRAL + SPIRAL_FACTOR * dt
        # compute angular speed considering that the linear speed is constant
        angular_speed = FORWARD_SPEED / radius
        # make agent move in spiral
        agent.set_velocity(FORWARD_SPEED, angular_speed)


class GoBackState(State):
    def __init__(self):
        super().__init__("GoBack")
        self.number_executions = 0  # counts the number of executions of this state

    def check_transition(self, agent, state_machine):
        # compute current time
        dt = self.number_executions * SAMPLE_TIME

        # check if agent spent enough time going back
        if dt > GO_BACK_TIME:
            state_machine.change_state(RotateState())

    def execute(self, agent):
        self.number_executions += 1

        # make agent go back
        agent.set_velocity(BACKWARD_SPEED, 0)


class RotateState(State):
    def __init__(self):
        super().__init__("Rotate")
        self.number_executions = 0

        # compute random dtheta in [-PI, PI[ that the agent will rotate
        dtheta = -math.pi + random.random() * 2 * math.pi

        # compute time to rotate dtheta with constant angular speed
        # and invert rotation direction if theta is negative
        if dtheta < 0:
            self.rotate_time = -dtheta / ANGULAR_SPEED
            self.angular_speed = -ANGULAR_SPEED
        else:
            self.rotate_time = dtheta / ANGULAR_SPEED
            self.angular_speed = ANGULAR_SPEED

    def check_transition(self, agent, state_machine):
        # compute current time
        dt = self.number_executions * SAMPLE_TIME
        # check if agent spent enough time rotating
        if dt > self.rotate_time:
            state_machine.change_state(MoveForwardState())

    def execute(self, agent):
        self.number_executions += 1

        # make agent rotate
        agent.set_velocity(0, self.angular_speed)
