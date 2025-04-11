function uploadImage(event) {
    event.preventDefault();
    const input = document.getElementById('imageInput');
    const file = input.files[0];
    if (!file) return alert("Selecciona una imagen");

    const formData = new FormData();
    formData.append("file", file);

    fetch("http://localhost:8000/predict", {
        method: "POST",
        body: formData
    })
        .then(response => response.json())
        .then(result => {
            document.getElementById("result").innerHTML = `
            <h2 class="text-primary">Resultado Clasificación</h2>
            <p><strong>Clase:</strong> ${result.class}</p>
            <p><strong>Confianza:</strong> ${result.confidence}%</p>
            <div class="row justify-content-center">
                <div class="col-md-6 mb-4">
                    <div class="card shadow">
                        <div class="card-header fw-bold">Imagen Original</div>
                        <div class="card-body img-container">
                            <img src="data:image/jpeg;base64,${result.original_image}" class="img-fluid rounded img-big">
                        </div>
                    </div>
                </div>
                <div class="col-md-6 mb-4">
                    <div class="card shadow">
                        <div class="card-header fw-bold">GradCAM</div>
                        <div class="card-body img-container">
                            <img src="data:image/jpeg;base64,${result.gradcam_image}" class="img-fluid rounded img-big">
                        </div>
                    </div>
                </div>
            </div>
        `;
        })
        .catch(error => {
            console.error("Error al clasificar la imagen:", error);
            alert("Ocurrió un error al clasificar la imagen.");
        });
}

function uploadImageD(event) {
    event.preventDefault();
    const input = document.getElementById('imageInput');
    const file = input.files[0];
    if (!file) return alert("Selecciona una imagen");

    const formData = new FormData();
    formData.append("file", file);

    fetch("http://localhost:8000/detect", {
        method: "POST",
        body: formData
    })
        .then(response => response.json())
        .then(result => {
            document.getElementById("result").innerHTML = `
            <h2 class="text-success">Resultado Detección</h2>
            <div class="row justify-content-center">
                <div class="col-md-6 mb-4">
                    <div class="card shadow justify-content-center">
                        <div class="card-header fw-bold">Imagen Original</div>
                        <div class="card-body img-container">
                            <img src="data:image/jpeg;base64,${result.original_image}" class="img-fluid rounded img-big">
                        </div>
                    </div>
                </div>
                <div class="col-md-6 mb-4">
                    <div class="card shadow justify-content-center">
                        <div class="card-header fw-bold">Detección</div>
                        <div class="card-body img-container">
                            <img src="data:detection_image/jpeg;base64,${result.detection_image}" class="img-fluid rounded img-big">
                        </div>
                    </div>
                </div>
            </div>
        `;
        })
        .catch(error => {
            console.error("Error al detectar la imagen:", error);
            alert("Ocurrió un error al detectar la imagen.");
        });
}

function uploadImageA(event) {
    event.preventDefault();
    const input = document.getElementById('imageInput');
    const file = input.files[0];
    if (!file) return alert("Selecciona una imagen");

    const formData = new FormData();
    formData.append("file", file);

    fetch("http://localhost:8000/detect", {
        method: "POST",
        body: formData
    })
        .then(response => response.json())
        .then(result => {
            document.getElementById("result").innerHTML = `
            <h2 class="text-success">Resultado Detección</h2>
            <div class="row justify-content-center">
                <div class="col-md-6">
                    <div class="card shadow">
                        <div class="card-header fw-bold">Detección</div>
                        <div class="card-body img-container">
                            <img src="data:detection_image/jpeg;base64,${result.detection_image}" class="img-fluid rounded">
                        </div>
                    </div>
                </div>
            </div>
        `;
        })
        .catch(error => {
            console.error("Error al detectar la imagen:", error);
            alert("Ocurrió un error al detectar la imagen.");
        });
}

function uploadImageZ(event) {
    event.preventDefault();
    const input = document.getElementById('imageInput');
    const file = input.files[0];
    if (!file) return alert("Selecciona una imagen");

    const formData = new FormData();
    formData.append("file", file);

    fetch("http://localhost:8000/predict", {
        method: "POST",
        body: formData
    })
        .then(response => response.json())
        .then(result => {
            document.getElementById("resultDetect").innerHTML = `
            <h2 class="text-primary">Resultado Clasificación</h2>
            <p><strong>Clase:</strong> ${result.class}</p>
            <p><strong>Confianza:</strong> ${result.confidence}%</p>
            <div class="row justify-content-center">
                <div class="col-md-4 mb-3">
                    <div class="card shadow">
                        <div class="card-header fw-bold">Imagen Original</div>
                        <div class="card-body img-container">
                            <img src="data:image/jpeg;base64,${result.original_image}" class="img-fluid rounded">
                        </div>
                    </div>
                </div>
                <div class="col-md-4 mb-3">
                    <div class="card shadow">
                        <div class="card-header fw-bold">GradCAM</div>
                        <div class="card-body img-container">
                            <img src="data:image/jpeg;base64,${result.gradcam_image}" class="img-fluid rounded">
                        </div>
                    </div>
                </div>
            </div>
            </div>
        `;
        })
        .catch(error => {
            console.error("Error al detectar la imagen:", error);
            alert("Ocurrió un error al detectar la imagen.");
        });
}

function init() {
    console.log("Frontend initialized.....");
    const randomNumber = Math.floor(Math.random() * 101); // número entre 0 y 100
    console.log("Número aleatorio:", randomNumber);
}
