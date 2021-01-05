var startClickConnect = function startClickConnect() {
    var clickConnect = function clickConnect() {
        console.log("Connnect Clicked");
        document.querySelector("#top-toolbar > colab-connect-button").shadowRoot.querySelector("#connect").click();
        document.querySelector("#top-toolbar > colab-connect-button").shadowRoot.querySelector("#connect").click();
    };
    click_interval = 300000;
    var intervalId = setInterval(clickConnect, click_interval);
    var intervalId2;
    setTimeout(() => { intervalId2 = setInterval(clickConnect, click_interval); }, 1000);
    var stopClickConnectHandler = function stopClickConnect() {
        clearInterval(intervalId);
        clearInterval(intervalId2);
        console.log("Stopped");
    };
    return stopClickConnectHandler;
};
var stopClickConnect = startClickConnect();