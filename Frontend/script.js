window.onload = function() {
  document.getElementById('fileInput').addEventListener('change', updateSelectedImage, false)
}

let inputImageName = ""

function updateSelectedImage(event) {
  var file = event.target.files[0]

  let reader = new FileReader()
    reader.onloadend = function() {
      document.getElementById("inputImageDisplay").src = reader.result
  }
  reader.readAsDataURL(file)

  inputImageName = file.name;
}

function submitRequest() {
  if (inputImageName == "") {
    alert("File not selected!")
    return
  }

  var formData = new FormData();
  formData.append('files', document.getElementById('fileInput').files[0]);

  axios({
      method: 'post',
      url: '/upload',
      data: formData,
      headers: {
          "content-type": "multipart/form-data"
      }
  }).then(function (response) {
      let xhttp = new XMLHttpRequest()
      xhttp.onload = function() {
        document.getElementById("outputImageDisplay").src = xhttp.responseText
      }
      xhttp.open("GET", "/poke?inputFile=" + inputImageName + "&drawBoundingBox=" + document.getElementById("boundingBoxOption").checked, false)
      xhttp.send()
  });
}