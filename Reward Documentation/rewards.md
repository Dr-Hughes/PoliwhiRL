The current reward goals in the default config are:

```json
    [[9, 1, 6, 50]],
    [[8, 4, 6, 50], [9, 4, 6, 50]],
    [[8, 5, 6, 50], [9, 5, 6, 50]],
    [[13, 6, 4, 50]],
    [[6, 4, 4, 50]],
    [[59, 8, 3, 50], [59, 9, 3, 50]],
    [[31, 13, 3, 50]]
```

These are read as 

x, y, map_location, reward

Where x and y are the coordinates of the reward, map_location is the map the
reward is on, and reward is the amount of reward given (though this isn't
currently used and all rewards are equal).


## Location details 

### 6 The Player's Downstairs House

This is the players downstairs house when first entering

There is a reward for entering this room for the first time from the starting
area 

| Location | Image |
|----------|-------|
| x: 9, y: 1, location: 6, map_num_loc: 6 | ![Image](./Image%20Examples/x_9_y_1_location_6_map_num_loc_6.png) |

Next we give two options, both trigger talking to the mother 

 Location | Image |
|----------|-------|
| x: 8, y: 4, location: 6, map_num_loc: 6 | ![Image](./Image%20Examples/x_8_y_4_location_6_map_num_loc_6.png) |
| x: 9, y: 4, location: 6, map_num_loc: 6 | ![Image](./Image%20Examples/x_9_y_4_location_6_map_num_loc_6.png) |


Finally in this room we reward for progressing past this point

| Location | Image |
|----------|-------|
| x: 8, y: 5, location: 6, map_num_loc: 6 | ![Image](./Image%20Examples/x_8_y_5_location_6_map_num_loc_6.png) |
| x: 9, y: 5, location: 6, map_num_loc: 6 | ![Image](./Image%20Examples/x_9_y_5_location_6_map_num_loc_6.png) |


### 4 Overworld of the first town

This is outside 

The first reward here is just entering

| Location | Image |
|----------|-------|
| x: 13, y: 6, location: 4, map_num_loc: 4 | ![Image](./Image%20Examples/x_13_y_6_location_4_map_num_loc_4.png) | 

And then the next is for reaching the door of the professor's lab 

| Location | Image |
|----------|-------|
| x: 6, y: 4, location: 4, map_num_loc: 4 | ![Image](./Image%20Examples/x_6_y_4_location_4_map_num_loc_4.png) | 

### 3 Overworld, first route out of town 

This is the first route out of town so we reward for reaching it, it means the
player also has a pokemon and is first off on their adventure to see Mr. Pokemon
/ Oak

| Location | Image |
|----------|-------|
| x: 59, y: 8, location: 4, map_num_loc: 3 | ![Image](./Image%20Examples/x_59_y_8_location_4_map_num_loc_3.png) |
| x: 59, y: 9, location: 4, map_num_loc: 3 | ![Image](./Image%20Examples/x_59_y_9_location_4_map_num_loc_3.png) |

We want to reward for making it through the tricky gap around here too 

| Location | Image |
|----------|-------|
| x: 31, y: 13, location: 4, map_num_loc: 3 | ![Image](./Image%20Examples/x_31_y_13_location_4_map_num_loc_3.png) |
### 

## Unused Locations

- 0 
  - This is a strange location that appears in memory when entering special
    scenes
- 5 This is the professors lab
- 7 Players bedroom
- 8 Random house in the first town
- 9 Another random house in the first town

## Misc Notes

- Map Loc and location are different! This is important to untangle at some
  point
- Map num loc is the most useful currently!