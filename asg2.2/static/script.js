const production_url = 'https://flaskapp-m4jbzqtwiq-uc.a.run.app/api/classify'
const development_url = 'http://127.0.0.1:5000/api/classify'

async function upload_and_classify(image) {
    console.log("Upload and classify called!");
    const data = await fetch(development_url, {
        method: 'POST',
        body: JSON.stringify({"image_b64": image}),
        headers: {
            'Content-Type': 'application/json'
        }
    }).then(async (response) => {
        const res = await response.json();
        // console.log('res ' + res)
        return res;
    }).catch((err) => {
        console.log(err);
    });
    const res_b64 = data['classified_b64'];
    const innerHTML = `<p><h2> Result: </h2></p>
                       <img src="data:image/png;base64,${res_b64}" alt="Classified Image">`;
    document.getElementById('predictedImage').innerHTML = innerHTML;
    document.getElementById('hasUploaded1').remove();
    document.getElementById('hasUploaded2').remove();
    $('#loadingModal').modal("hide");
    
}