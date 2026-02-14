# Lib Neural

- json_loader
  - Acts as a dummy loader to load the JOSN from file into some data sturctures
- model 
  - Handles the actual math and logic of the neural netowrok
  - Torch stores Params into JSON like  `[IN][OUT]` = `[2][3]`
  - The math library wants Matrix<3, 2>: `[OUT][IN]` = `[3][2]` for simple dot products
- types
