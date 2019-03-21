# Gym Example for thegame

## Installation

thegame's server requires go1.11+ and client requires python3.6+. Their dependencies are listed below:

### Python

```
pip install -r requirements.txt
```

or

```
pip install --upgrade git+https://github.com/openai/{baselines,gym}
pip install --upgrade git+https://github.com/afg984/thegame.git#subdirectory=client/python
```

### Go

```
go get -u github.com/afg984/thegame/server/go/thegame
```

## Run

```
python -m baselines.run --extra_import gym_thegame --env=thegame-v0 --alg=ppo2 --network=lstm --num_timesteps=2e9 --nsteps=2048 --lr=4e-3 --gamma=0.999 --nminibatches=1
```

## Spectate

Something like this

```
python -m thegame.gui.spectator localhost:50051
```

localhost:50051 is the default address. If you change it in `TheGameEnv`, you should spectate on
the address you set as well.
