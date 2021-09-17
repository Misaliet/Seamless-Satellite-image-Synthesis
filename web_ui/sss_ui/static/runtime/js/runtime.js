// Use leaflet.js?

function init() {
    // var large = document.getElementById("fourteen wide column hide");
    // large.style.display = "none";
    var obj = document.getElementById("map");
    // console.log(obj)
    drag(obj);
}

window.onload = init;

function test() {
    alert("test!!!");
}

var level = 0;
function get_z() {
    var httpRequest = new XMLHttpRequest();
    httpRequest.onreadystatechange = function () {
        if (httpRequest.readyState == 4 && httpRequest.status == 200) {
            level = parseInt(httpRequest.responseText);
        }
    };
    httpRequest.open('GET', '/get_z', false);
    httpRequest.send();
}

function random() {
    // alert("random!!!");
    var httpRequest = new XMLHttpRequest();
    httpRequest.open('GET', '/random', true);
    httpRequest.send();

    httpRequest.onreadystatechange = function () {
        if (httpRequest.readyState == 4 && httpRequest.status == 200) {
            // alert("left!");
            var mapIMG = document.getElementById("map")
            var indexQ = mapIMG.src.lastIndexOf("?")
            if (indexQ == -1) {
                indexQ = mapIMG.src.length
            }
            var pathA = mapIMG.src.substring(0, indexQ)
            pathA = pathA + "?" + Math.random()
            mapIMG.src = pathA
        }
    };
}

function restore() {
    // alert("restore!!!");
    var httpRequest = new XMLHttpRequest();
    httpRequest.open('GET', '/restore', true);
    httpRequest.send();

    httpRequest.onreadystatechange = function () {
        if (httpRequest.readyState == 4 && httpRequest.status == 200) {
            // alert("left!");
            var mapIMG = document.getElementById("map")
            var indexQ = mapIMG.src.lastIndexOf("?")
            if (indexQ == -1) {
                indexQ = mapIMG.src.length
            }
            var pathA = mapIMG.src.substring(0, indexQ)
            pathA = pathA + "?" + Math.random()
            mapIMG.src = pathA
        }
    };
}

function left() {
    var httpRequest = new XMLHttpRequest();
    httpRequest.open('GET', '/left', true);
    httpRequest.send();

    httpRequest.onreadystatechange = function () {
        if (httpRequest.readyState == 4 && httpRequest.status == 200) {
            // alert("left!");
            var mapIMG = document.getElementById("map")
            var indexQ = mapIMG.src.lastIndexOf("?")
            if (indexQ == -1) {
                indexQ = mapIMG.src.length
            }
            var pathA = mapIMG.src.substring(0, indexQ)
            pathA = pathA + "?" + Math.random()
            mapIMG.src = pathA
        }
    };
}

function up() {
    var httpRequest = new XMLHttpRequest();
    httpRequest.open('GET', '/up', true);
    httpRequest.send();

    httpRequest.onreadystatechange = function () {
        if (httpRequest.readyState == 4 && httpRequest.status == 200) {
            // alert("up!");
            var mapIMG = document.getElementById("map")
            var indexQ = mapIMG.src.lastIndexOf("?")
            if (indexQ == -1) {
                indexQ = mapIMG.src.length
            }
            var pathA = mapIMG.src.substring(0, indexQ)
            pathA = pathA + "?" + Math.random()
            mapIMG.src = pathA
        }
    };
}

function down() {
    var httpRequest = new XMLHttpRequest();
    httpRequest.open('GET', '/down', true);
    httpRequest.send();

    httpRequest.onreadystatechange = function () {
        if (httpRequest.readyState == 4 && httpRequest.status == 200) {
            // alert("down!");
            var mapIMG = document.getElementById("map")
            var indexQ = mapIMG.src.lastIndexOf("?")
            if (indexQ == -1) {
                indexQ = mapIMG.src.length
            }
            var pathA = mapIMG.src.substring(0, indexQ)
            pathA = pathA + "?" + Math.random()
            mapIMG.src = pathA
        }
    };
}

function right() {
    var httpRequest = new XMLHttpRequest();
    httpRequest.open('GET', '/right', true);
    httpRequest.send();

    httpRequest.onreadystatechange = function () {
        if (httpRequest.readyState == 4 && httpRequest.status == 200) {
            // alert("right!");
            var mapIMG = document.getElementById("map")
            var indexQ = mapIMG.src.lastIndexOf("?")
            if (indexQ == -1) {
                indexQ = mapIMG.src.length
            }
            var pathA = mapIMG.src.substring(0, indexQ)
            pathA = pathA + "?" + Math.random()
            mapIMG.src = pathA
        }
    };
}

function transfer() {
    // alert("transfer!!!");
    get_z();
    // alert(level);
    if (level > 1) {
        alert("Can't transfer at current level!")
        return;
    }
    var httpRequest = new XMLHttpRequest();
    httpRequest.open('GET', '/transfer', true);
    httpRequest.send();

    httpRequest.onreadystatechange = function () {
        if (httpRequest.readyState == 4 && httpRequest.status == 200) {
            alert("transfer!");
            // map is not changed
            // var mapIMG = document.getElementById("map")
            // console.log(mapIMG.naturalWidth)
            // console.log(mapIMG.naturalHeight)
            // var path1 = mapIMG.src + "?" + Math.random()
            // console.log(path1);
            // mapIMG.src = path2

            var satelliteIMG = document.getElementById("satellite")
            var indexQ = satelliteIMG.src.lastIndexOf("?")
            if (indexQ == -1) {
                indexQ = satelliteIMG.src.length
            }
            var pathB = satelliteIMG.src.substring(0, indexQ)
            pathB = pathB + "?" + Math.random()
            satelliteIMG.src = pathB
        }
    };
}

function zoomIn() {
    var httpRequest = new XMLHttpRequest();
    httpRequest.open('GET', '/zoom_in', true);
    httpRequest.send();

    httpRequest.onreadystatechange = function () {
        if (httpRequest.readyState == 4 && httpRequest.status == 200) {
            alert("zoom_in!");
            var mapIMG = document.getElementById("map")
            var indexQ = mapIMG.src.lastIndexOf("?")
            if (indexQ == -1) {
                indexQ = mapIMG.src.length
            }
            var pathA = mapIMG.src.substring(0, indexQ)
            pathA = pathA + "?" + Math.random()
            mapIMG.src = pathA

            var satelliteIMG = document.getElementById("satellite")
            var indexQ = satelliteIMG.src.lastIndexOf("?")
            if (indexQ == -1) {
                indexQ = satelliteIMG.src.length
            }
            var pathB = satelliteIMG.src.substring(0, indexQ)
            pathB = pathB + "?" + Math.random()
            satelliteIMG.src = pathB
        }
    };
}

function superResolution() {
    get_z();
    // alert(level);
    if (level < 2) {
        alert("Can't match at current level!")
        return;
    }
    var httpRequest = new XMLHttpRequest();
    httpRequest.open('GET', '/super_resolution', true);
    httpRequest.send();

    httpRequest.onreadystatechange = function () {
        if (httpRequest.readyState == 4 && httpRequest.status == 200) {
            alert("match!");
            var satelliteIMG = document.getElementById("satellite")
            var indexQ = satelliteIMG.src.lastIndexOf("?")
            if (indexQ == -1) {
                indexQ = satelliteIMG.src.length
            }
            var pathB = satelliteIMG.src.substring(0, indexQ)
            pathB = pathB + "?" + Math.random()
            satelliteIMG.src = pathB
        }
    };
}

function zoomOut() {
    var httpRequest = new XMLHttpRequest();
    httpRequest.open('GET', '/zoom_out', true);
    httpRequest.send();

    httpRequest.onreadystatechange = function () {
        if (httpRequest.readyState == 4 && httpRequest.status == 200) {
            alert("zoom_out!");
            var mapIMG = document.getElementById("map")
            var indexQ = mapIMG.src.lastIndexOf("?")
            if (indexQ == -1) {
                indexQ = mapIMG.src.length
            }
            var pathA = mapIMG.src.substring(0, indexQ)
            pathA = pathA + "?" + Math.random()
            mapIMG.src = pathA
        }
    };
}


// FIXME: think of a better way to handle mouse wheel events
// var doScroll = true;

// if (window.addEventListener)
//     window.addEventListener('DOMMouseScroll', wheel, false);
// window.onmousewheel = document.onmousewheel = wheel;

// function wheel(event){
//     if (detectArea(event)) {
//         event.preventDefault();
//         event.stopPropagation();
//         if (doScroll) {
//             doScroll = false;
//             var delta = 0;
//             if (!event) event = window.event;
//             if (event.wheelDelta) {
//                 delta = event.wheelDelta/120; 
//                 if (window.opera) delta = -delta;
//             } else if (event.detail) {
//                 delta = -event.detail/3;
//             }
//             if (delta) {
//                 // setTimeout(handle,500,delta); 
//                 handle(delta);
//             }
//         }
//         return false;
//     } else {
//         return false;
//     }
// }

function handle(delta) {
    var large = document.getElementById("fourteen wide column hide");
    if (large.style.display == "none") {
        if (delta <0){
            // zoom out
            zoomOut();
            // doScroll = true;
        }else{
            // zoom in
            // alert("!!!")
            zoomIn();
            // doScroll = true;
        }
    }
}

var moveWheel1 = true;
var moveWheel2 = false;
var wheelClock;
function stopWheel(event) {
    if (moveWheel2 == true) {
        // console.log("stop!");
        moveWheel2 = false;
        moveWheel1 = true;
        //
        if (detectArea(event)) {
            var delta = 0;
            if (!event) event = window.event;
            if (event.wheelDelta) {
                delta = event.wheelDelta/120; 
                if (window.opera) delta = -delta;
            } else if (event.detail) {
                delta = -event.detail/3;
            }
            if (delta) {
                // setTimeout(handle,500,delta); 
                handle(delta);
            }
        }
    }
}
function moveWheel(event) {
    if (moveWheel1==true) {
        // console.log("start!");
        moveWheel1 = false;
        moveWheel2 = true;
        wheelClock = setTimeout(stopWheel, 200, event);
    }
    else {
        clearTimeout(wheelClock);
        wheelClock = setTimeout(stopWheel, 150, event);
    }
}

document.addEventListener('wheel', moveWheel, false);

function detectArea(e) {
    var div = document.getElementById("map");
    var x = e.clientX;
    var y = e.clientY;
    var divx1 = getLeft(div);
    var divy1 = getTop(div);
    var divx2 = divx1 + div.offsetWidth;
    var divy2 = divy1 + div.offsetHeight;
    // console.log(x,y,divx1,divy1,divx2,divy2)
    if( x < divx1 || x > divx2 || y < divy1 || y > divy2){
        return false;
    } else {
        return true;
    }
}

function getTop(e) {
    var offset=e.offsetTop;
    if (e.offsetParent!=null) offset+=getTop(e.offsetParent);
    return offset;
}

function getLeft(e) {
    var offset=e.offsetLeft;
    if (e.offsetParent!=null) offset+=getLeft(e.offsetParent);
    return offset;
}


// drag
function drag(obj) {
    obj.addEventListener('mousedown', start);
    function start(event) {
        if (event.button == 0) {
            offsetX = event.pageX - obj.offsetLeft + parseInt(getComputedStyle(obj)['margin-left']);
            offsetY = event.pageY - obj.offsetTop + parseInt(getComputedStyle(obj)['margin-top']);
            // console.log(offsetX, offsetY);
            document.addEventListener('mousemove', move);
            document.addEventListener('mouseup', stop);
        }
        return false;
    }

    function move(event) {
        // obj.style.left = (event.pageX - offsetX) + 'px';
        // obj.style.top = (event.pageY - offsetY) + 'px';
        // console.log(event.pageX - offsetX, event.pageY - offsetY);
        if ((event.pageX - offsetX) > 0 && (event.pageX - offsetX)%8 == 0) {
            left();
        } else if ((offsetX - event.pageX) > 0 && (offsetX - event.pageX)%8 == 0) {
            right();
        }

        if ((event.pageY - offsetY) > 0 && (event.pageY - offsetY)%8 == 0) {
            up();
        } else if ((offsetY - event.pageY) > 0 && (offsetY - event.pageY)%8 == 0) {
            down();
        }

        return false;
    }

    function stop(envet) {
        // console.log(2);
        document.removeEventListener('mousemove', move);
        document.removeEventListener('mouseup', stop);
    }
}

function showLarge() {
    var large = document.getElementById("fourteen wide column hide");
    if (large.style.display == "block")
        large.style.display = "none";
    else
        large.style.display = "block";

    var largeImage = document.getElementById("large");
    var indexQ = largeImage.src.lastIndexOf("?")
    if (indexQ == -1) {
        indexQ = largeImage.src.length
    }
    var pathA = largeImage.src.substring(0, indexQ)
    pathA = pathA + "?" + Math.random()
    largeImage.src = pathA
}

/*
GET:
var httpRequest = new XMLHttpRequest();
httpRequest.open('POST', 'url', true);
httpRequest.setRequestHeader("Content-type","application/x-www-form-urlencoded");
httpRequest.send('name=teswe&ee=ef');
httpRequest.onreadystatechange = function () {
    if (httpRequest.readyState == 4 && httpRequest.status == 200) {
        var json = httpRequest.responseText;
        console.log(json);
    }
};

POST:
var httpRequest = new XMLHttpRequest();
httpRequest.open('POST', 'url', true);
httpRequest.setRequestHeader("Content-type","application/x-www-form-urlencoded");
httpRequest.send('name=teswe&ee=ef');
httpRequest.onreadystatechange = function () {
    if (httpRequest.readyState == 4 && httpRequest.status == 200) {
        var json = httpRequest.responseText;
        console.log(json);
    }
};

POST-json:
var httpRequest = new XMLHttpRequest();
httpRequest.open('POST', 'url', true);
httpRequest.setRequestHeader("Content-type","application/json");
httpRequest.send(JSON.stringify(obj));
httpRequest.onreadystatechange = function () {
    if (httpRequest.readyState == 4 && httpRequest.status == 200) {
        var json = httpRequest.responseText;
        console.log(json);
    }
};
*/
