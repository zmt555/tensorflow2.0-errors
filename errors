{
   1.GpuLaunchKernel
   GpuLaunchKernel(FillPhiloxRandomKernelLaunch<Distribution>, num_blocks, block_size, 0, d.stream(), gen, data, size, dist) status: Internal: invalid device function
   solution:
   When I was configuring the environment, I installed cuda10.1. 
   When I uninstalled cuda10.1 and installed cuda10.0, this problem was solved.
}
{
   2.cuDNN failed to initialize
   UnknownError: Failed to get convolution algorithm. This is probably because cuDNN failed to initialize, so try looking to see if a warning log message was printed above. [Op:Conv2D]
   solution:
   This error may be caused by a graphics memory explosion. So we need to limit the graphics memory. Just add the following code to the program.
   
   gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
   tf.config.experimental.set_memory_growth(gpus[0], True)
   tf.config.experimental.set_virtual_device_configuration(
   	gpus[0],
    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5000)]  # 6G memory, so I use 5000 
}
