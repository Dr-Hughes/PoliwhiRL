"""The goal ladder. Success is judged from RAM by the harness — the model is
never trusted to declare it. Goal text is phrased as in-game knowledge (what
the game's characters and signs tell a player), never as coordinates.

Flag numbers are event-flag bits from 0xDA72 (docs/reference/ram-mapping.md,
PoliwhiRL/checkpoints.py).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

TALKED_TO_MOM = 1735  # derived-table flag, set during Mom's Pokégear scene
GOT_STARTER = 26  # EVENT_GOT_A_POKEMON_FROM_ELM
GOT_MYSTERY_EGG = 30
GAVE_MYSTERY_EGG = 31
BEAT_FALKNER = 1213


@dataclass(frozen=True)
class Goal:
    key: str
    title: str
    text: str
    done: Callable  # (game, vars) -> bool
    budget: int  # env steps per attempt
    optional: bool = False  # on failure, restore the start state and carry on
    # (game, vars) -> str: what the RAM check is measuring, shown every turn
    # with "not complete yet" so the model can't talk itself into success.
    check: Callable | None = None


def _on_map(bank, num):
    return lambda g, v: (int(v["map_bank"]), int(v["map_num"])) == (bank, num)


def _flag_and_free(n):
    return lambda g, v: g.flag(n, v) and not v["script_active"]


LADDER = (
    Goal("stairs", "Go downstairs",
         "Go down the stairs from your bedroom to the floor below.",
         _on_map(24, 6), 80),
    Goal("mom", "Talk to Mom",
         "Talk to Mom downstairs and get through the whole conversation, "
         "until you can walk around freely again.",
         _flag_and_free(TALKED_TO_MOM), 200),
    Goal("outside", "Leave the house",
         "Leave the house and step outside into the town.",
         _on_map(24, 4), 120),
    Goal("starter", "Get a starter Pokémon",
         "Get your first Pokémon from Professor Elm, whose lab is in this town. "
         "The goal is complete once a Pokémon is in your party and the scene has ended.",
         lambda g, v: g.flag(GOT_STARTER, v) and g.party_count() >= 1 and not v["script_active"],
         700),
    Goal("route29", "Reach Route 29",
         "Professor Elm asked you to visit Mr. Pokémon, who lives past Cherrygrove City. "
         "Leave New Bark Town by walking off its WEST edge onto Route 29.",
         _on_map(24, 3), 250),
    Goal("cherrygrove", "Reach Cherrygrove City",
         "Travel west along Route 29 until you reach Cherrygrove City.",
         _on_map(26, 3), 900),
    Goal("mr_pokemon", "Get the Mystery Egg from Mr. Pokémon",
         "Mr. Pokémon's house is north of Cherrygrove City along Route 30. Go there, talk to "
         "Mr. Pokémon and receive what he has for Professor Elm. The goal is complete when "
         "you have it and the scene has ended.",
         _flag_and_free(GOT_MYSTERY_EGG), 1800),
    Goal("back_to_elm", "Return the egg to Elm",
         "Bring the Mystery Egg back to Professor Elm's lab in New Bark Town (south along "
         "Route 30 to Cherrygrove City, then east along Route 29). Talk to Elm and give it to "
         "him. Someone may challenge you to a battle on the way: fight it.",
         _flag_and_free(GAVE_MYSTERY_EGG), 3000),
    Goal("catch", "Catch a wild Pokémon",
         "Catch a wild Pokémon. You need Poké Balls (Elm's assistant gives you some after "
         "you return the egg; Poké Marts sell them). Walk through tall grass to meet wild "
         "Pokémon, weaken one with attacks, then choose PACK in battle and throw a Poké Ball.",
         lambda g, v: int(v["pokedex_owned"]) >= 2 and not v["script_active"],
         2000, optional=True,
         check=lambda g, v: f"Pokémon you own: {int(v['pokedex_owned'])} (needs 2: one you "
                            "catch yourself with your own Poké Ball)"),
    Goal("violet", "Reach Violet City",
         "Travel to Violet City: north from Cherrygrove City along Route 30, then west along "
         "Route 31; a gatehouse at its end leads into Violet City. If your Pokémon are hurt, "
         "heal them by talking to the nurse in a Pokémon Center.",
         _on_map(10, 5), 4000),
    Goal("violet_gym", "Enter the Violet Gym",
         "Find the Violet City Pokémon Gym and go inside.",
         _on_map(10, 7), 800),
    Goal("falkner", "Beat Falkner",
         "Defeat Falkner, the Violet Gym Leader, at the back of the gym; gym trainers may "
         "battle you on the way. If your Pokémon faint or are low on HP, heal at the Violet "
         "City Pokémon Center and come back. Training on wild Pokémon in tall grass first "
         "makes the fight easier.",
         _flag_and_free(BEAT_FALKNER), 5000),
)
GOALS = {g.key: g for g in LADDER}
