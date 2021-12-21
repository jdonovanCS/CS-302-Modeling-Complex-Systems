// box counting
var initiate_box_counting = false;
var box_count_start_r = 10;
var boxes_drawn = true;
var box_count_results = [];
// gui params
var gui;
var mu_prev;
var max_particles_prev;
var start_stop=true;
var levi_flight=false;
// grid params
var grid_size_sq = 50;
var sq_size_px = 10;
var grid_size_px = grid_size_sq * sq_size_px;
var grid = []
// colors
var background_color;
var particle_color;
var structure_color;
// particles
var max_particles = 1;
// levy flight
var mu = 2.0;
var max_step_size = grid_size_sq / 2;

class Particle {
  constructor() {
    // init particle on grid
    this.restart();
    // init probs
    this.calc_probs();
  }

  move() {
    // direction of the movement
    this.x_direction = 0;
    this.y_direction = 0;
    while (this.x_direction == 0 && this.y_direction == 0)
      {
        this.x_direction = int(random([-1,0,1]));
        this.y_direction = int(random([-1,0,1]));
      }
    if(levi_flight)
    {
       // get step size from a power distribution parametrized by mu
      this.r = this.get_step_size(); 
    }
    else
    {
      this.r = 1;
    }
    // make that step
    this.x += this.x_direction * this.r;
    this.y += this.y_direction * this.r;
    this.x = cc(this.x);
    this.y = cc(this.y);
  }
  
  get_step_size(){
    let rand = random();
    var sum = 0;
    for (let i=0; i<max_step_size; i++)
      {
        sum += this.probs[i];
        if ( rand < sum )
          {
            return i+1;
          }
      }
    return this.probs.length;
  }
  
  calc_probs()
  {
    // set probabilities
    this.probs = [];
    var sum = 0;
    for (let i=0; i<max_step_size; i++)
      {
        this.probs[i] = 1.0 / ((i+1)**mu);
        sum += 1.0 / ((i+1)**mu);
      }
    for (let i=0; i<max_step_size; i++)
      {
        this.probs[i] = this.probs[i] / sum;
      }
  }
  
  restart()
  {
    let try_again = true;
    while (try_again)
      {
        this.x = int(random(grid_size_sq));
        this.y = int(random(grid_size_sq));
        if ( grid[this.x][this.y] == 0 && 
            !check_neighbors(this.x,this.y)[0] )
          {
            try_again = false;
          }
      }
    grid[this.x][this.y] = 1;
  }
  
}
var particles = [];



function setup() {

  createCanvas(windowWidth, windowHeight);

  // Create the GUI
  gui = createGui('have fun!').setPosition(0, 0);
  sliderRange(0.75, 3, 0.01);
  gui.addGlobals('mu');
  sliderRange(1, 50, 1);
  gui.addGlobals('max_particles');
  gui.addGlobals('levi_flight');
  //gui.addGlobals('start_stop');
  restart_button = new Clickable();//Create button
  restart_button.locate(0, windowHeight-50);//Position Button
  restart_button.text = "restart";//Text of the clickable
  restart_button.onPress = function()
  {  //When myButton is pressed
    if(initiate_box_counting)
    {
      initiate_box_counting = false;
      start_stop_button.text = "stop";//Text of the clickable
      start_stop = true;
    }
    // restart
    for (var x = 0; x<grid_size_sq; x++)
    {
      for (var y = 0; y<grid_size_sq; y++)
      {
        if(grid[x][y] == -1)
        {
          grid[x][y] = 0;
        }
      }
    }
    grid[int(grid_size_sq/2)][int(grid_size_sq/2)] = -1;
    draw_grid();
    draw_grid();
    draw_grid();
  }
  ss_button = new Clickable();//Create button
  ss_button.locate(100, windowHeight-50);//Position Button
  ss_button.text = "screenshot";//Text of the clickable
  ss_button.onPress = function()
  {  //When myButton is pressed
    // take a screenshot
    saveCanvas('canvas.png');
  }
  start_stop_button = new Clickable();//Create button
  start_stop_button.locate(200, windowHeight-50);//Position Button
  start_stop_button.text = "stop";//Text of the clickable
  start_stop_button.onPress = function()
  {  //When myButton is pressed
    if(start_stop_button.text == " "){return;}
    // starts or stops
    if(start_stop == true)
    {
      start_stop_button.text = "start";//Text of the clickable
      start_stop = false;
    }
    else
    {
      start_stop_button.text = "stop";//Text of the clickable
      start_stop = true;
    }
  }
  init_box_count_button = new Clickable();//Create button
  init_box_count_button.locate(300, windowHeight-50);//Position Button
  init_box_count_button.text = "init box counting";//Text of the clickable
  init_box_count_button.onPress = function()
  {  //When myButton is pressed
   // initiates box counting
    initiate_box_counting = true;
    do_box_counting();
  }
  draw_box_button = new Clickable();//Create button
  draw_box_button.locate(300, windowHeight-50);//Position Button
  draw_box_button.text = "toggle boxes";//Text of the clickable
  draw_box_button.onPress = function()
  {  //When myButton is pressed
    // toggles boxes
    if(boxes_drawn)
    {
      draw_grid();
      boxes_drawn = false;
    }
    else
    {
      draw_grid();
      do_box_counting(); 
      boxes_drawn = true;
    }
  }
  save_results_button = new Clickable();//Create button
  save_results_button.locate(400, windowHeight-50);//Position Button
  save_results_button.text = "save results";
  save_results_button.onPress = function()
  {  //When myButton is pressed
    // create txt file with the box counting results
    if(initiate_box_counting == false)
    {
      alert("count them boxes first");
    }
    else
    {
      var to_write = [];
      for(var i=1; i<box_count_results.length; i++)
      {
        to_write[i-1] = String(i) + " " + String(box_count_results[i]);
      }
      save(to_write, 'results.txt');
    }
    
  }
  
  // color settings
  background_color = color(200,200,200,80);
  particle_color = color(50,50,50);
  structure_color = color(250,0,0);

  
  //init grid
  init_grid()
  
  // init particles
  for (let i=0; i<max_particles; i++)
    {
      particles[i] = new Particle();
    }

  // control gui parameters 
  mu_prev = mu;
  max_particles_prev = max_particles;
}


function draw() {
  
  if (frameCount % 1 == 0 && start_stop && initiate_box_counting == false)
    {
      background(0,0,0,120);
      draw_grid();
      update_grid();
    }
  
  if (mu != mu_prev)
    {
      // mu has changed
      for (let i=0; i<particles.length; i++)
        {
          particles[i].calc_probs();
        }
      mu_prev = mu;
    }
  if (max_particles != max_particles_prev)
    {
      // max_particles has changed
      if (max_particles < max_particles_prev)
        {
          var to_pop = max_particles_prev - max_particles;
          for (let i=0;i<to_pop;i++)
            {
              var to_delete = particles.pop();
              grid[to_delete.x][to_delete.y] = 0;
            }
        }
      else if (max_particles > max_particles_prev)
        {
          var to_push = max_particles - max_particles_prev;
          for (let i=0;i<to_push;i++)
            {
              particles.push( new Particle() );
            }
        }
      max_particles_prev = max_particles;
    }
  restart_button.draw();// restarts the structure
  ss_button.draw();// takes a screenshot
  start_stop_button.draw();// plays or pauses
  init_box_count_button.draw();// initiates box counting
  if(initiate_box_counting)
  {
    draw_box_button.draw();
  }
  save_results_button.draw();
}


// dynamically adjust the canvas to the window
function windowResized() {
  resizeCanvas(windowWidth, windowHeight);
}

// draw the grid structure 
function draw_grid()
{
  var start_x = (windowWidth/2)-(grid_size_px/2);
  var start_y = (windowHeight/2)-(grid_size_px/2);
  for (let x=0; x<grid_size_sq; x++)
    {
      for (let y=0; y<grid_size_sq; y++)
        {
          //background
          if (grid[x][y] == 0)
            {
              fill(background_color);
              stroke(background_color);
            }
          // particle
          else if ( grid[x][y] == 1 )
            {
              fill(particle_color);
              stroke(particle_color);
            }
          // structure
          else if ( grid[x][y] == -1 )
            {
              fill(structure_color);
              stroke(structure_color);
            }
          // calc squares position
          var curr_x = start_x + x*sq_size_px;
          var curr_y = start_y + y*sq_size_px;
          // draw the actual square
          strokeWeight(1);
          square(curr_x, curr_y, sq_size_px);
        }
    }
}

function init_grid()
{
  for (let i=0; i<grid_size_sq; i++)
    {
      grid[i] = [];
      for (let j=0; j<grid_size_sq; j++)
        {
          grid[i][j] = 0;
        }
    }
  grid[int(grid_size_sq/2)][int(grid_size_sq/2)] = -1;
}

function update_grid()
{
  for (let i=0; i<particles.length; i++)
    {
      // store previous location
      prev_x = particles[i].x;
      prev_y = particles[i].y;
      // let particle try to move
      particles[i].move();
      // prev location will be emptied no matter what
      grid[prev_x][prev_y] = 0;
      // control the path
var agg = control_aggregation(prev_x,prev_y,particles[i].x_direction,particles[i].y_direction, particles[i].r)
      if (agg) // particle becomes part of the structure
        {
          // re init the particle
          particles[i].restart();
        }
      else // move particle to next position
        {
          next_x = particles[i].x;
          next_y = particles[i].y;  
          grid[next_x][next_y] = 1;
        }
    }
}

function control_aggregation(start_x, start_y, direction_x, direction_y, step_size)
{
  var agg = false;
  var x = start_x;
  var y = start_y;
  for (let i=0; i<step_size; i++)
  {
    prev_x = x;
    prev_y = y;
    x += direction_x;
    y += direction_y;
    if(x == cc(x) && y == cc(y))
    {
      draw_path(prev_x,prev_y,x,y,step_size);
    }
    x = cc(x);
    y = cc(y);
    var results = check_neighbors(x,y);
    if ( results[0] )
    {
      // aggregate
      grid[x][y] = -1;
      agg = true;
      // check whether to initiate box counting directly
      if(x==0 || y==0 || x==grid_size_sq-1 || y==grid_size_sq-1)
      {
        initiate_box_counting = true;
        do_box_counting();
      }
      return agg;
    }
  }
  return agg;
}

function draw_path(start_x, start_y, end_x, end_y, step_size)
{
  var leftmost = (windowWidth/2)-(grid_size_px/2) + int(sq_size_px/2);
  var topmost = (windowHeight/2)-(grid_size_px/2) + int(sq_size_px/2);
  start_x = leftmost + start_x*sq_size_px;
  start_y = topmost + start_y*sq_size_px;
  end_x = leftmost + end_x*sq_size_px;
  end_y = topmost + end_y*sq_size_px;
  push();
  var redness = 50 + (150/max_step_size)*step_size;
  stroke(color(redness,0,0));
  if(step_size<7)
  {
   strokeWeight(0); 
  }
  else if(step_size<13)
  {
   strokeWeight(1); 
  }
  else
  {
   strokeWeight(2); 
  }
  line(start_x,start_y,end_x,end_y);
  pop();
}
function check_neighbors(x,y)
{
  // sol ust
  if (grid[ cc(x-1) ][ cc(y-1) ] == -1)
    {
      return [true, cc(x-1), cc(y-1)];
    }
  // ust
  else if (grid[ cc(x-1) ][ cc(y) ] == -1)
    {
      return [true, cc(x-1), cc(y) ];
    }
  // sag ust
  else if (grid[ cc(x-1) ][ cc(y+1) ] == -1)
    {
      return [true, cc(x-1), cc(y+1) ];
    }
  // sag
  else if (grid[ cc(x) ][ cc(y+1) ] == -1)
    {
      return [true, cc(x), cc(y+1)];
    }
  // sag alt
  else if (grid[cc(x+1)][cc(y+1)] == -1)
    {
      return [true, cc(x+1), cc(y+1)];
    }
  // alt
  else if (grid[cc(x+1)][cc(y)] == -1)
    {
      return [true, cc(x+1), cc(y)];
    }
  // sol alt
  else if (grid[cc(x+1)][cc(y-1)] == -1)
    {
      return [true, cc(x+1), cc(y-1)];
    }
  // sol
  else if (grid[cc(x)][cc(y-1)] == -1)
    {
      return [true, cc(x), cc(y-1)];
    }
  // no aggregate
  else
    {
      return [false, -1, -1];
    }
}

function cc(x) //correct coordinate
{
  return (x+grid_size_sq)%grid_size_sq;
}

function do_box_counting()
{
  // set the states right
  start_stop_button.text = " ";//Text of the clickable
  start_stop = false;
  // get rid of unnecessary things on the grid
  for (let x=0; x<grid_size_sq; x++)
  {
    for (let y=0; y<grid_size_sq; y++)
    {
      if( grid[x][y] != -1)
      {
        grid[x][y] = 0;
      }
    }
  }
  // do counting for differently sized boxes
  box_count_results = [];
  for(var i=box_count_start_r; i>=1; i--)
  {
    var count = convolve(i);
    console.log(i, count);
    box_count_results[i] = count;
  }
  // print results on screen
  var rightmost = (windowWidth/2)+(grid_size_px/2)+50;
  var topmost = (windowHeight/2)-(grid_size_px/2);
  textSize(24);
  text('r', rightmost+10, topmost);
  text('N', rightmost+50, topmost);
  for(var i=1; i<=box_count_start_r; i++)
  {
    text(String(i), rightmost+10, topmost+(i)*50);
    text(String(box_count_results[i]), rightmost+50, topmost+(i)*50);
  }
}
function convolve(kernel_size)
{
  var total = 0;
  var to_break = false;
  for (let x_start=0; x_start<grid_size_sq; x_start+=kernel_size)
  {
    for (let y_start=0; y_start<grid_size_sq; y_start+=kernel_size)
    {
      for (let x=x_start; x<x_start+kernel_size; x++)
      {
        if(to_break){to_break=false;break;}
        for (let y=y_start; y<y_start+kernel_size; y++)
        {
          if(x>=grid_size_sq || y>=grid_size_sq){continue;}
          if(grid[x][y] == -1)
          {
            total++;
            draw_box(x_start,y_start,kernel_size);
            to_break=true;
            break;
          }
        }
        if(to_break&&x==x_start+kernel_size-1){to_break=false;}
      }
    }
  }
  return total;
}

function draw_box(x_s,y_s,kernel_size)
{
  var leftmost = (windowWidth/2)-(grid_size_px/2);
  var topmost = (windowHeight/2)-(grid_size_px/2);
  
  push();
  strokeWeight(kernel_size);
  r = random(255); // r is a random number between 0 - 255
  g = random(100,200); // g is a random number betwen 100 - 200
  b = random(100); // b is a random number between 0 - 100
  stroke(color(r,g,b));
  noFill();   square(leftmost+x_s*sq_size_px,topmost+y_s*sq_size_px,kernel_size*sq_size_px);
  pop();
  
  
}