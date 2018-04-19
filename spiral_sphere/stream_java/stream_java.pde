
  
BufferedReader reader;
String line;
boolean init;
void setup(){
  size(800, 800,P3D);
  beginCamera();
  camera();
  camera(0, 0, 300, 0, 0, 0, 0, 1, 0);
  endCamera();
  
  reader = createReader("../data_stream");
  line = "";
  background(0);
  stroke(255);

  init = false;
}

void draw() { 
  print("draw()");
  if(!init){
    init = true;
    return;
  }
    
  line = "";
  
  try {
    while(true){
      line = reader.readLine();
      String[] pieces = split(line, " ");
      if(pieces[0].equals("draw")){
        break;
      }
      else if(pieces[0].equals("points")){
        int num_points = int(pieces[1]);
        println("Drawing "+num_points+" points\n");
        for(int i = 0; i < num_points; i++){
          line = reader.readLine();
          String[] xy = split(line, " ");
          float x = float(xy[0]);
          float y = float(xy[1]);
          //print("New point: "+x+", "+y+"\n");
          point(x*100, y*100);
        }
      }
      else if(pieces[0].equals("clear")){
        background(0);
      }
    }
  } catch (IOException e) {
    e.printStackTrace();
    line = null;
  }
  if (line == null) {
    // Stop reading because of an error or file is empty
    noLoop();  
  } else {
  }
  println("Done drawing");
} 

//finally {
//  try {
//    input.close();
//  } 
//  catch (IOException e) {
//    e.printStackTrace();
//  }
//}
