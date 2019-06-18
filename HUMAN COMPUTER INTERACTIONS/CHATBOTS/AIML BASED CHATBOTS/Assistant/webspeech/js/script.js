/*
webspeech
*/
try {
  var SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  var recognition = new SpeechRecognition();
}
catch(e) {
  console.error(e);
  alert("browser not supported or no mic found !");
}
recognition.continuous = true;
recognition.onresult = function(event) {
  var data = $('#data');
  var current = event.resultIndex;
  var transcript = event.results[current][0].transcript;
  console.log("transcript : " + transcript);
  data.val(transcript);
};

recognition.onerror = function(event) {
  if(event.error == 'no-speech') {
    console.log("error");
  };
}

recognition.start();
