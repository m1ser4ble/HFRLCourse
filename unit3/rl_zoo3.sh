#!/bin/bash

# train
python -m rl_zoo3.train --algo dqn --env SpaceInvadersNoFrameskip-v4 -f logs/ -c dqn.yaml
# eval
python -m rl_zoo3.enjoy --algo dqn --env SpaceInvadersNoFrameskip-v4 --n-timesteps 5000 --folder logs/
# upload
python -m rl_zoo3.push_to_hub --algo dqn --env SpaceInvadersNoFrameskip-v4 --repo-name dqn-SpaceInvadersNoFrameskip-v4 -orga hung3r -f logs/
