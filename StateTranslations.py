import pandas as pd


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
    # return [1, 0, 1, 0, -0.007170/(cart_vel + pole_angle + pole_ang_vel), 0]


def translate_mountaincar_to_cartpole(mountaincart_state):
    """
    Translates a mountaincar state to an cartpole state, so the
    pre-trained cartpole model can extract some meaning from it.
    :param mountaincart_state: [Car Position, Car Velocity]
    :return: cartpole state: [Cart Position, Cart Velocity, Pole Angle, Pole Angular Velocity]
    """
    car_pos, car_vel = mountaincart_state
    return [0, 0, 0, 1/car_vel]
    # return [0, 0, 0, 0.000076/car_vel]


def translate_cartpole_to_mountaincar(cartpole_state):
    """
    Translates a cartpole state to an mountaincar state, so the
    pre-trained acrobot model can extract some meaning from it.
    :param cartpole_state: [Cart Position, Cart Velocity, Pole Angle, Pole Angular Velocity]
    :return: mountaincart_state: [Car Position, Car Velocity]
    """
    cart_pos, cart_vel, pole_angle, pole_ang_vel = cartpole_state
    return [0, -1/(cart_vel + pole_angle + pole_ang_vel)]
    # return [0, -0.036016/(cart_vel + pole_angle + pole_ang_vel)]


def translate_mountaincar_to_acrobot(mountaincart_state):
    """
    Translates a mountaincar state to an cartpole state, so the
    pre-trained cartpole model can extract some meaning from it.
    :param mountaincart_state: [Car Position, Car Velocity]
    :return: cartpole state: [Cart Position, Cart Velocity, Pole Angle, Pole Angular Velocity]
    """
    car_pos, car_vel = mountaincart_state
    return [1, 0, 1, 0, car_vel, 0]
    # return [1, 0, 1, 0, 0.199078*car_vel, 0]


state_translations = {
    'CartPole-v1>Acrobot-v1': translate_cartpole_to_acrobot,
    'MountainCarContinuous-v0>CartPole-v1': translate_mountaincar_to_cartpole,
    'CartPole-v1>MountainCarContinuous-v0': translate_cartpole_to_mountaincar,
    'MountainCarContinuous-v0>Acrobot-v1': translate_mountaincar_to_acrobot,
}

if __name__ == "__main__":

    # script for finding out how to scale translated states to comply with the mean and stdev known to pre-trained agent
    env_names = {'CartPole-v1': 'cartpole', 'Acrobot-v1': 'acrobot', 'MountainCarContinuous-v0': 'mountaincar'}
    drop_cols = ['ep', 'step', 'action']
    for source_env_name, source_nickname in env_names.items():
        source_samples = pd.read_csv(f'hist_{source_env_name}.csv').drop(columns=drop_cols)
        source_cols = list(source_samples.columns)
        df_source = pd.DataFrame(source_samples.mean(), columns=['source mean'])
        # df_source['source std'] = source_samples.std()
        for target_env_name, target_nickname in env_names.items():
            translation_name = f'{target_env_name}>{source_env_name}'
            if translation_name in state_translations:
                print(f'\n{translation_name}:')
                translate = state_translations[translation_name]
                target_samples = pd.read_csv(f'hist_{target_env_name}.csv').drop(columns=drop_cols)
                translated_samples = [translate(row) for i, row in target_samples.iterrows()]
                translated_samples = pd.DataFrame(translated_samples, columns=source_cols)
                df = df_source.copy()
                df['translated mean'] = translated_samples.mean()
                # df['translated std'] = translated_samples.std()
                c1, c2 = 'source mean', 'translated mean'
                df['scale translation by'] = [r[c1]/r[c2] if r[c2] != 0 else 1 for i, r in df.iterrows()]
                print(f'{df}')
