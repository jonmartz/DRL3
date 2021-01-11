import math

def translate_cartpole_to_acrobot(cartpole_state):
    """
    Translates a cartpole state to an acrobot state, so the
    pre-trained acrobot model can extract some meaning from it.
    :param cartpole_state: [Cart Position, Cart Velocity, Pole Angle, Pole Angular Velocity]
    :return: acrobot state: [cos(theta1), sin(theta1), cos(theta2), sin(theta2), thetaDot1, thetaDot2]

             where theta1 and theta2 are the angles of the inner and outer joints and thetaDot_i is angular velocity
    """
    cart_pos, cart_vel, pole_angle, pole_ang_vel = cartpole_state
    return [1, 0, 1, 0, -1/(cart_vel + pole_angle + pole_ang_vel), 0]


def translate_mountaincar_to_cartpole(mountaincart_state):
    """
    Translates a mountaincar state to an cartpole state, so the
    pre-trained cartpole model can extract some meaning from it.
    :param mountaincart_state: [Car Position, Car Velocity]
    :return: cartpole state: [Cart Position, Cart Velocity, Pole Angle, Pole Angular Velocity]
    """
    car_pos, car_vel = mountaincart_state
    return [0, 0, 0, 1/car_vel]


def translate_cartpole_to_mountaincar(cartpole_state):
    """
    Translates a cartpole state to an mountaincar state, so the
    pre-trained acrobot model can extract some meaning from it.
    :param cartpole_state: [Cart Position, Cart Velocity, Pole Angle, Pole Angular Velocity]
    :return: mountaincart_state: [Car Position, Car Velocity]
    """
    cart_pos, cart_vel, pole_angle, pole_ang_vel = cartpole_state
    return [0, -1/(cart_vel + pole_angle + pole_ang_vel)]


def translate_mountaincar_to_acrobot(mountaincart_state):
    """
    Translates a mountaincar state to an cartpole state, so the
    pre-trained cartpole model can extract some meaning from it.
    :param mountaincart_state: [Car Position, Car Velocity]
    :return: cartpole state: [Cart Position, Cart Velocity, Pole Angle, Pole Angular Velocity]
    """
    car_pos, car_vel = mountaincart_state
    return [1, 0, 1, 0, car_vel, 0]


state_translations = {
    'CartPole-v1>Acrobot-v1': translate_cartpole_to_acrobot,
    'MountainCarContinuous-v0>CartPole-v1': translate_mountaincar_to_cartpole,
    'CartPole-v1>MountainCarContinuous-v0': translate_cartpole_to_mountaincar,
    'MountainCarContinuous-v0>Acrobot-v1': translate_mountaincar_to_acrobot,
}
