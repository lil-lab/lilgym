from gymnasium.envs.registration import register


register(
    id='TowerScratch-v0',
    entry_point='lilgym.envs:NaturalLanguageVisualReasoningEnv',
    kwargs={
        'appearance': 'tower',
        'starting_condition': 'scratch',
    },
)

register(
    id='TowerFlipIt-v0',
    entry_point='lilgym.envs:NaturalLanguageVisualReasoningEnv',
    kwargs={
        'appearance': 'tower',
        'starting_condition': 'flipit',
    },
)

register(
    id='ScatterScratch-v0',
    entry_point='lilgym.envs:NaturalLanguageVisualReasoningEnv',
    kwargs={
        'appearance': 'scatter',
        'starting_condition': 'scratch',
    },
)

register(
    id='ScatterFlipIt-v0',
    entry_point='lilgym.envs:NaturalLanguageVisualReasoningEnv',
    kwargs={
        'appearance': 'scatter',
        'starting_condition': 'flipit',
    },
)