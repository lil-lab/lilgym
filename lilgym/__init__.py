from gym.envs.registration import register


register(
    id='TowerScratch-v0',
    entry_point='lilgym.envs:NaturalLanguageVisualReasoningEnv',
    kwargs={
        'env_opt': 'tower',
        'learn_opt': 'scratch',
    },
)

register(
    id='TowerFlipIt-v0',
    entry_point='lilgym.envs:NaturalLanguageVisualReasoningEnv',
    kwargs={
        'env_opt': 'tower',
        'learn_opt': 'flipit',
    },
)

register(
    id='ScatterScratch-v0',
    entry_point='lilgym.envs:NaturalLanguageVisualReasoningEnv',
    kwargs={
        'env_opt': 'scatter',
        'learn_opt': 'scratch',
    },
)

register(
    id='ScatterFlipIt-v0',
    entry_point='lilgym.envs:NaturalLanguageVisualReasoningEnv',
    kwargs={
        'env_opt': 'scatter',
        'learn_opt': 'flipit',
    },
)