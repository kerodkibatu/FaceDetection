using static Emgu.CV.CvInvoke;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;

CascadeClassifier model = new CascadeClassifier("haarcascade_frontalface_default.xml");
VideoCapture vc = new(0);
while (true) 
{
    Mat frame = new();
    vc.Read(frame);
    //Resize(frame, frame,new Size(),0.5,0.5);

    Mat gray = new();
    CvtColor(frame, gray, ColorConversion.Rgb2Gray);

    var result = model.DetectMultiScale(gray,minNeighbors: 10);

    foreach (var bounds in result)
    {
        Rectangle(frame, bounds, new MCvScalar(0, 0, 255), 5);
    }

    Imshow("My Window",frame);
    if (WaitKey(1) == 65)
        break;
}