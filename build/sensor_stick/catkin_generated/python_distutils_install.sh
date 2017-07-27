#!/bin/sh

if [ -n "$DESTDIR" ] ; then
    case $DESTDIR in
        /*) # ok
            ;;
        *)
            /bin/echo "DESTDIR argument must be absolute... "
            /bin/echo "otherwise python's distutils will bork things."
            exit 1
    esac
    DESTDIR_ARG="--root=$DESTDIR"
fi

echo_and_run() { echo "+ $@" ; "$@" ; }

echo_and_run cd "/home/robond/perception/src/sensor_stick"

# snsure that Python install destination exists
echo_and_run mkdir -p "$DESTDIR/home/robond/perception/install/lib/python2.7/dist-packages"

# Note that PYTHONPATH is pulled from the environment to support installing
# into one location when some dependencies were installed in another
# location, #123.
echo_and_run /usr/bin/env \
    PYTHONPATH="/home/robond/perception/install/lib/python2.7/dist-packages:/home/robond/perception/build/lib/python2.7/dist-packages:$PYTHONPATH" \
    CATKIN_BINARY_DIR="/home/robond/perception/build" \
    "/usr/bin/python" \
    "/home/robond/perception/src/sensor_stick/setup.py" \
    build --build-base "/home/robond/perception/build/sensor_stick" \
    install \
    $DESTDIR_ARG \
    --install-layout=deb --prefix="/home/robond/perception/install" --install-scripts="/home/robond/perception/install/bin"
