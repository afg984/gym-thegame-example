# Gym Example for thegame

## Installation

python3.6+

```
pip install --upgrade git+https://github.com/openai/{baselines,gym}
pip install --upgrade git+https://github.com/afg984/thegame.git#subdirectory=client/python
```

go

```
go get -u github.com/afg984/thegame/server/go/thegame
```

## Configuration

Update `bin=...` in gym_thegame.py with $GOPATH/bin/thegame

## Run

```
python -m baselines.run --extra_import gym_thegame --env=thegame-v0 --alg=ppo2 --network=lstm --num_timesteps=2e9 --nsteps=2048 --lr=4e-3 --gamma=0.999 --num_env=4
```

## Spectate

Something like this

```
python -m thegame.gui.spectator localhost:50051
```
