function predict() {

    document.getElementById("loading").style.display = "block";
    document.getElementById("result").innerText = "";
    document.getElementById("lrBar").style.width = "0%";
    document.getElementById("rfBar").style.width = "0%";

    const data = {
        year: Number(document.getElementById("year").value),
        km_driven: Number(document.getElementById("km").value),
        fuel: document.getElementById("fuel").value,
        seller_type: document.getElementById("seller").value,
        transmission: document.getElementById("transmission").value,
        owner: Number(document.getElementById("owner").value)
    };

    fetch("http://127.0.0.1:8000/predict?model=rf", {
        method: "POST",
        headers: {"Content-Type":"application/json"},
        body: JSON.stringify(data)
    })
    .then(res => res.json())
    .then(result => {

        document.getElementById("loading").style.display = "none";

        if (result.error) {
            document.getElementById("result").innerText = result.error;
        } else {
            document.getElementById("result").innerText =
                "Estimated Price: " + result.predicted_price;

            // 🎯 Animate bars
            document.getElementById("lrBar").style.width = "84%";
            document.getElementById("rfBar").style.width = "88%";
        }

    })
    .catch(() => {
        document.getElementById("loading").style.display = "none";
        document.getElementById("result").innerText = "API Error!";
    });
}