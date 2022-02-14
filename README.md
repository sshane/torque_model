# torque_model

The torque model is a spiritual successor to [op-smart-torque](https://github.com/sshane/op-smart-torque), which was a project to train a neural network to control a car's steering fully end to end.

The input is the current wheel angle and future wheel angle ([among other things](https://github.com/sshane/torque_model/blob/103b5ddca091dcfabf8002f2c820515024f77acf/lib/helpers.py#L17)), and the net's output is the torque to apply to the wheel to reach that smoothly and confidently. This bypasses the need to manually tune a PID, LQR, or INDI controller, while gaining human-like control over the steering wheel.

Needs to be cloned into an [openpilot](https://github.com/commaai/openpilot) repo to take advantage of its tools.

# The problem

As talked about in great detail and with a simple thought experiment in [comma.ai](https://comma.ai)'s blog post [here](https://blog.comma.ai/end-to-end-lateral-planning/) about end to end lateral planning, the same concept of behavior cloning not being able to recover from disturbances applies here.

## Behavior cloning and lack of perturbations

The way we generate automatically-labeled training data for a model that predicts how to control a steering wheel is rather simple; any time a human is driving we just take the current (t<sub>0s</sub>) and future (t<sub>0.5s</sub>) steering wheel angles and then just have the model predict whatever torque the human was applying at t<sub>0s</sub>.

This seems to works great, and the validation loss also seems to be really low! However, when you actually try to drive on this model or put it in a simulator, you can quickly see that any small disturbances (like wind, road camber, etc) quickly lead to a feedback loop or just plain inability to correct back to our desired steering angle.

This is due to the automatically-generated training and validation data containing only samples where the current and future (desired during runtime) steering wheel angles are very close together (just a couple degrees), as a symptom of only using data where the future angle is just fractions of a second away.

To fully realize the problem, think about what would happen if you wanted this model to predict what a human would actuate if the steering wheel is centered, but our desired angle is something like 90 degrees. As the model has never seen a difference of angles higher than just a couple of degrees, it either outputs a very small torque value, or just nonsense, as this input is vastly outside of its training distribution.

# The solution

The solution talked about in the blog post above is to use a very simple simulator to tell the model what path the human drove and then warp the input video to be offset left or right, as well as introducing oscillation. This approach can also be taken here, where we generate random samples with an arbitrary steering wheel angle error, and then use a simple model of predicting torque, like a PF (proportional-feedforward) controller as the output to predict.

For the example above where we start at 0 degrees and want to reach 90 degrees, we can inject samples into the training data where we have that exact situation and then have the output be what a simple PF controller would output. Then during runtime in the car, when the model corrects for this arbitrary high angle error situation, the current and desired steering wheel angles become much closer together, and the model can then use its knowledge of how humans control under these circumstances.
