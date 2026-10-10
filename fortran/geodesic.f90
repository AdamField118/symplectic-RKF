! Bernard Ymeri's symplectic RK4/RK6 geodesic integrator (WPI MQP 2026,
! Appendix A), ported to Fortran.
!
! Each function matches the Python one of the same name, and the comments in
! them are Bernard's, copied as written. Arrays are 1-based here, so Python's
! g[0, 0] is g(1, 1) and y[:n] is y(1:n). Everything is double precision.
!
! The command line and the output are the same as the GLSL version's: the
! result is a NumPy .npy file of shape (n_geodesics, steps + 1, 8), one row of
! (t, r, theta, phi, p_t, p_r, p_theta, p_phi) per step. Each geodesic is
! independent, so they're spread over OpenMP threads (OMP_NUM_THREADS, or all
! cores). A single geodesic is a chain of steps that each depend on the last,
! so it runs on one core.
!
!   ./geodesic [--M 1.0] [--a 0.5] [--h 1] [--steps 15000] [--chunk 100]
!              [--ic "t r th ph pt pr pth pph"]... [--ic-file FILE] [--out geodesic.npy]

module geodesic_integrator
    use, intrinsic :: iso_fortran_env, only: dp => real64
    implicit none

    ! Kerr metric set up
    real(dp) :: M = 1.0_dp
    real(dp) :: a = 0.5_dp

    ! Symplectic RK4 coefficients
    real(dp), parameter :: r4a11 = 1.0_dp/4
    real(dp), parameter :: r4a12 = 1.0_dp/4 - sqrt(3.0_dp)/6
    real(dp), parameter :: r4a21 = 1.0_dp/4 + sqrt(3.0_dp)/6
    real(dp), parameter :: r4a22 = 1.0_dp/4

    real(dp), parameter :: r4b1 = 1.0_dp/2
    real(dp), parameter :: r4b2 = 1.0_dp/2

    ! Symplectic RK6 coefficients
    real(dp), parameter :: r6a11 = 5.0_dp/36
    real(dp), parameter :: r6a12 = 2.0_dp/9 - sqrt(15.0_dp)/15
    real(dp), parameter :: r6a13 = 5.0_dp/36 - sqrt(15.0_dp)/30

    real(dp), parameter :: r6a21 = 5.0_dp/36 + sqrt(15.0_dp)/24
    real(dp), parameter :: r6a22 = 2.0_dp/9
    real(dp), parameter :: r6a23 = 5.0_dp/36 - sqrt(15.0_dp)/24

    real(dp), parameter :: r6a31 = 5.0_dp/36 + sqrt(15.0_dp)/30
    real(dp), parameter :: r6a32 = 2.0_dp/9 + sqrt(15.0_dp)/15
    real(dp), parameter :: r6a33 = 5.0_dp/36

    real(dp), parameter :: r6b1 = 5.0_dp/18
    real(dp), parameter :: r6b2 = 4.0_dp/9
    real(dp), parameter :: r6b3 = 5.0_dp/18

    ! The right-hand side f(y) that the RK steps are handed (hamiltonian here)
    abstract interface
        function rhs(y) result(dydt)
            import :: dp
            real(dp), intent(in) :: y(:)
            real(dp) :: dydt(size(y))
        end function rhs
    end interface

contains

    ! Inverse Kerr metric (uses the inverse metric like the Fantasty code). Takes in
    !a vector of length 4 representing the 4 Boyder-Lindquist coordinates: t, r,
    !theta, and phi
    function inverse_metric(x) result(g)
        real(dp), intent(in) :: x(4)
        real(dp) :: g(4, 4)
        real(dp) :: t, r, theta, phi, Sigma, Delta

        t = x(1)
        r = x(2)
        theta = x(3)
        phi = x(4)

        !Making it easier and more compact to write out the full Kerr metric
        Sigma = r**2 + a**2 * cos(theta)**2
        Delta = r**2 - 2*M*r + a**2

        !Set up a 4x4 matrix of zeroes
        g = 0.0_dp

        !Input all the spacetime metric components (Kerr for this example)
        g(1, 1) = -((r**2 + a**2)**2 - a**2 * Delta * sin(theta)**2) / (Delta * Sigma)
        g(1, 2) = 0.0_dp
        g(1, 3) = 0.0_dp
        g(1, 4) = -2 * M * a * r / (Delta * Sigma)

        g(2, 1) = g(1, 2)
        g(2, 2) = Delta / Sigma
        g(2, 3) = 0.0_dp
        g(2, 4) = 0.0_dp

        g(3, 1) = g(1, 3)
        g(3, 2) = g(2, 3)
        g(3, 3) = 1.0_dp / Sigma
        g(3, 4) = 0.0_dp

        g(4, 1) = g(1, 4)
        g(4, 2) = g(2, 4)
        g(4, 3) = g(3, 4)
        g(4, 4) = (Delta - a**2 * sin(theta)**2) / (Delta * Sigma * sin(theta)**2)
    end function inverse_metric

    ! Numerical derivatives of inverse metric
    function dgdx(x, delta_in) result(dg)
        real(dp), intent(in) :: x(:)
        real(dp), intent(in), optional :: delta_in
        real(dp) :: dg(size(x), size(x), size(x))
        real(dp) :: delta, dx(size(x)), g_plus(4, 4), g_minus(4, 4)
        integer :: n, w

        delta = 1e-6_dp
        if (present(delta_in)) delta = delta_in

        ! Find the number of variables we're working with and create an n by n by n array
        !that will hold all the partial derivvatives of every entry of the inverse metric
        !matrix
        n = size(x)
        dg = 0.0_dp

        ! Uses central finite differences to approximate each partial derivative.
        !The n-th loop computes all 4 partial derivatives for the n-th spacetime
        !coordinate.
        do w = 1, n
            dx = 0.0_dp
            dx(w) = delta
            g_plus = inverse_metric(x + dx)
            g_minus = inverse_metric(x - dx)
            dg(w, :, :) = (g_plus - g_minus) / (2 * delta)
        end do
    end function dgdx

    ! Hamiltonian
    function hamiltonian(y) result(dydt)
        real(dp), intent(in) :: y(:)
        real(dp) :: dydt(size(y))
        real(dp) :: x(size(y) / 2), p(size(y) / 2), g(4, 4), dg(4, 4, 4)
        real(dp) :: dx(size(y) / 2), dp_(size(y) / 2)
        integer :: n, w, u, v, j

        ! Takes the first n elements of the initial conditions vector y (these
        !correspond to the position coordinates) and the last n elements of y (these
        !correspond to the momentum coordinates), and makes them their own vectors.
        n = size(y) / 2
        x = y(1:n)
        p = y(n + 1:)

        g = inverse_metric(x)
        dg = dgdx(x)

        ! Following the Hamiltonian equations for geodesics, we calculate the
        !derivatives of the position and momentum components
        ! (dx = np.dot(g,p), summed in order)
        do w = 1, n
            dx(w) = 0.0_dp
            do j = 1, n
                dx(w) = dx(w) + g(w, j) * p(j)
            end do
        end do

        dp_ = 0.0_dp

        ! Uses a triple for-loop to sum over the 3 indices in the equation for the
        !momentum derivative
        do w = 1, n
            do u = 1, n
                do v = 1, n
                    dp_(w) = dp_(w) - 0.5_dp * dg(w, u, v) * p(u) * p(v)
                end do
            end do
        end do

        dydt(1:n) = dx
        dydt(n + 1:) = dp_
    end function hamiltonian

    ! The 2-norm, as np.linalg.norm computes it: sqrt of the sum of squares
    pure function norm(v)
        real(dp), intent(in) :: v(:)
        real(dp) :: norm
        norm = sqrt(dot_product(v, v))
    end function norm

    ! Symplectic RK4 step, uses fixed point iteration to solve the implicit RK method
    function symplectic_rk4(f, y, h, tol_in, max_iter_in) result(y_next)
        procedure(rhs) :: f
        real(dp), intent(in) :: y(:), h
        real(dp), intent(in), optional :: tol_in
        integer, intent(in), optional :: max_iter_in
        real(dp) :: y_next(size(y))
        real(dp) :: tol, k1(size(y)), k2(size(y)), k1_new(size(y)), k2_new(size(y))
        integer :: max_iter, it

        tol = 5e-11_dp
        if (present(tol_in)) tol = tol_in
        max_iter = 20
        if (present(max_iter_in)) max_iter = max_iter_in

        ! Gives the initial guess for k1 and k2 based on the input
        k1 = f(y)
        k2 = f(y)

        ! Updates k1 and k2 based on the symplectic rk4 coefficients and the previous
        !guess for k1 and k2.
        do it = 1, max_iter
            k1_new = f(y + h * (r4a11 * k1 + r4a12 * k2))
            k2_new = f(y + h * (r4a21 * k1 + r4a22 * k2))

            ! Checks if the difference (in the regular 2-norm) of k1 and the updated
            !k1 is less than the given tolerance, and if the same is true for k2 and
            !the updated k2.
            if ((norm(k1_new - k1) < tol) .and. (norm(k2_new - k2) < tol)) then
                exit
            end if

            ! Replaces the previous guess with the new one
            k1 = k1_new
            k2 = k2_new
        end do

        ! Uses the final guesses and the symplectic rk4 coefficients to update y.
        y_next = y + h * (r4b1 * k1 + r4b2 * k2)
    end function symplectic_rk4

    ! Symplectic RK6 step, uses fixed point iteration to solve the implicit RK method
    function symplectic_rk6(f, y, h, tol_in, max_iter_in) result(y_next)
        procedure(rhs) :: f
        real(dp), intent(in) :: y(:), h
        real(dp), intent(in), optional :: tol_in
        integer, intent(in), optional :: max_iter_in
        real(dp) :: y_next(size(y))
        real(dp) :: tol, k1(size(y)), k2(size(y)), k3(size(y))
        real(dp) :: k1_new(size(y)), k2_new(size(y)), k3_new(size(y))
        integer :: max_iter, it

        tol = 5e-11_dp
        if (present(tol_in)) tol = tol_in
        max_iter = 20
        if (present(max_iter_in)) max_iter = max_iter_in

        ! Gives the initial guess for k1, k2, and k3 based on the input
        k1 = f(y)
        k2 = f(y)
        k3 = f(y)

        ! Checks if the difference (in the regular 2-norm) of k1 and the updated
        !k1 is less than the given tolerance, and if the same is true for k2 and
        !the updated k2, and k3 and the updated k3.
        do it = 1, max_iter
            k1_new = f(y + h * (r6a11*k1 + r6a12*k2 + r6a13*k3))
            k2_new = f(y + h * (r6a21*k1 + r6a22*k2 + r6a23*k3))
            k3_new = f(y + h * (r6a31*k1 + r6a32*k2 + r6a33*k3))

            ! Convergence check (2-norm)
            if (norm(k1_new - k1) < tol .and. &
                norm(k2_new - k2) < tol .and. &
                norm(k3_new - k3) < tol) then
                exit
            end if

            ! Replaces the previous guess with the new one
            k1 = k1_new
            k2 = k2_new
            k3 = k3_new
        end do

        ! Uses the final guesses and the symplectic rk6 coefficients to update y.
        y_next = y + h * (r6b1*k1 + r6b2*k2 + r6b3*k3)
    end function symplectic_rk6

    ! We'll compute an rk4 and an rk6 step, find the error between them, and decide
    !whether to accept the rk4 step, or shrink the step size and try again until the
    !error is less than a given tolerance.
    function symplectic_rk46_comparison(f, y, h, tol_in, max_shrinks_in, rk_tol_in, &
                                        rk_max_iter_in) result(y_rk4)
        procedure(rhs) :: f
        real(dp), intent(in) :: y(:), h
        real(dp), intent(in), optional :: tol_in, rk_tol_in
        integer, intent(in), optional :: max_shrinks_in, rk_max_iter_in
        real(dp) :: y_rk4(size(y))
        real(dp) :: tol, rk_tol, h_current, error, y_rk6(size(y))
        integer :: max_shrinks, rk_max_iter, it

        tol = 0.1_dp
        if (present(tol_in)) tol = tol_in
        max_shrinks = 1
        if (present(max_shrinks_in)) max_shrinks = max_shrinks_in
        rk_tol = 5e-11_dp
        if (present(rk_tol_in)) rk_tol = rk_tol_in
        rk_max_iter = 20
        if (present(rk_max_iter_in)) rk_max_iter = rk_max_iter_in

        ! Define the current default step size
        h_current = h

        ! A for loop = computes an rk4 and rk6 step until the error between them is
        !below a given tolerance, then computes them again with a smaller step size
        !if it's not
        do it = 1, max_shrinks + 1

            ! Compute a symplectic rk4 step
            y_rk4 = symplectic_rk4(f, y, h_current, tol_in=rk_tol, max_iter_in=rk_max_iter)

            ! Compute a symplectic rk6 step
            y_rk6 = symplectic_rk6(f, y, h_current, tol_in=rk_tol, max_iter_in=rk_max_iter)

            ! Find the 2 norm of their difference
            error = norm(y_rk6 - y_rk4)

            ! Accept symplectic rk4 step if error is below the tolerance
            if (error < tol) then
                return
            end if

            ! Update step size using formula that's used in the rkf method
            h_current = 0.9_dp * h_current * (tol / error)**(1.0_dp/5)
        end do

        ! If the error still isn't below the tolerance after max_shrinks, accept the rk4 step
    end function symplectic_rk46_comparison

    ! Integrate Geodesic
    ! (Python returns the whole list; here the caller hands in where to put the
    ! steps, so it can take them `chunk` at a time. Row 1 is the starting point.)
    subroutine integrate_geodesic(y, h, steps, geodesic)
        real(dp), intent(inout) :: y(:)
        real(dp), intent(in) :: h
        integer, intent(in) :: steps
        real(dp), intent(out) :: geodesic(size(y), steps)
        integer :: s

        ! Compute the position/momentum vector using the symplectic rk4, adds it to the
        !trajectory list, and then computes the next position/momentum in the geodesic
        do s = 1, steps
            y = symplectic_rk46_comparison(hamiltonian, y, h)
            geodesic(:, s) = y
        end do
    end subroutine integrate_geodesic

end module geodesic_integrator


program geodesic
    use, intrinsic :: iso_fortran_env, only: dp => real64, error_unit, int64
    use geodesic_integrator
    !$ use omp_lib
    implicit none

    real(dp) :: h = 1.0_dp
    integer :: steps = 15000, chunk = 100
    character(len=4096) :: out = 'geodesic.npy', ic_file = '', key, val
    character(len=32) :: elapsed
    real(dp), allocatable :: ics(:, :), traj(:, :, :), y(:, :)
    real(dp) :: v(8)
    integer :: i, nargs, n_ic, g, done, c, ios, u
    integer(int64) :: t0, t1, rate

    ! Command line, the same options as the GLSL version
    n_ic = 0
    allocate(ics(8, 0))
    nargs = command_argument_count()
    i = 1
    do while (i <= nargs)
        call get_command_argument(i, key)
        if (key == '-h' .or. key == '--help') then
            write(error_unit, '(a)') 'see the comment at the top of geodesic.f90 for options'
            stop
        end if
        if (i == nargs) call die('option without a value')
        call get_command_argument(i + 1, val)
        i = i + 2
        select case (trim(key))
        case ('--M')
            read(val, *) M
        case ('--a')
            read(val, *) a
        case ('--h')
            read(val, *) h
        case ('--steps')
            read(val, *) steps
        case ('--chunk')
            read(val, *) chunk
        case ('--out')
            out = val
        case ('--ic-file')
            ic_file = val
        case ('--ic')
            read(val, *, iostat=ios) v
            if (ios /= 0) call die('--ic needs 8 numbers')
            call add_ic(v)
        case default
            write(error_unit, '(2a)') 'unknown option ', trim(key)
            stop 1
        end select
    end do
    if (len_trim(ic_file) > 0) then
        open(newunit=u, file=trim(ic_file), status='old', action='read', iostat=ios)
        if (ios /= 0) call die('cannot open --ic-file')
        do
            read(u, *, iostat=ios) v
            if (ios /= 0) exit
            call add_ic(v)
        end do
        close(u)
    end if
    if (n_ic == 0) then
        ! Appendix A's initial conditions
        call add_ic([0.0_dp, 20.0_dp, 3.141592653589793_dp / 2, 0.0_dp, &
                     -0.9764550153430405_dp, 0.0_dp, 3.8_dp, 3.0_dp])
    end if
    if (chunk < 1) chunk = 1
    if (chunk > steps .and. steps > 0) chunk = steps

    !$ write(error_unit, '(a, i0)') 'OpenMP threads: ', omp_get_max_threads()

    ! traj(:, s, g) is row s of geodesic g, so in memory it's laid out exactly
    ! like the C-order (n_geodesics, steps + 1, 8) array the .npy file holds
    allocate(traj(8, steps + 1, n_ic), y(8, n_ic))
    y = ics(:, 1:n_ic)
    traj(:, 1, :) = y   ! geodesic.append(y.copy()) for the starting point

    call system_clock(t0, rate)
    done = 0
    do while (done < steps)
        c = min(chunk, steps - done)
        !$omp parallel do schedule(dynamic, 1)
        do g = 1, n_ic
            call integrate_geodesic(y(:, g), h, c, traj(:, done + 2:done + 1 + c, g))
        end do
        !$omp end parallel do
        done = done + c
        call system_clock(t1)
        write(elapsed, '(f0.1)') real(t1 - t0, dp) / rate
        if (elapsed(1:1) == '.') elapsed = '0' // trim(elapsed)   ! print 0.3 like the GLSL version, not .3
        write(error_unit, '(a, i0, a, i0, 3a)', advance='no') &
            char(13) // 'step ', done, '/', steps, '  ', trim(elapsed), ' s'
    end do

    call write_npy(trim(out), traj)
    write(error_unit, '(/, 3a, i0, a, i0, a)') 'wrote ', trim(out), '  shape (', n_ic, ', ', &
        steps + 1, ', 8)'

contains

    subroutine die(msg)
        character(*), intent(in) :: msg
        write(error_unit, '(2a)') 'error: ', msg
        stop 1
    end subroutine die

    subroutine add_ic(v)
        real(dp), intent(in) :: v(8)
        real(dp), allocatable :: tmp(:, :)
        if (n_ic == size(ics, 2)) then
            allocate(tmp(8, max(4, 2 * n_ic)))
            tmp(:, 1:n_ic) = ics(:, 1:n_ic)
            call move_alloc(tmp, ics)
        end if
        n_ic = n_ic + 1
        ics(:, n_ic) = v
    end subroutine add_ic

    ! The same .npy header the GLSL version writes, then the doubles
    subroutine write_npy(path, arr)
        character(*), intent(in) :: path
        real(dp), intent(in) :: arr(:, :, :)
        character(len=256) :: dict
        integer :: n, pad, hlen, u
        write(dict, '(a, i0, a, i0, a)') "{'descr': '<f8', 'fortran_order': False, 'shape': (", &
            size(arr, 3), ', ', size(arr, 2), ", 8), }"
        n = len_trim(dict)
        pad = modulo(64 - modulo(10 + n + 1, 64), 64)
        hlen = n + pad + 1
        open(newunit=u, file=path, access='stream', form='unformatted', status='replace', &
             action='write')
        write(u) char(147) // 'NUMPY' // char(1) // char(0) // char(modulo(hlen, 256)) // &
                 char(hlen / 256) // dict(1:n) // repeat(' ', pad) // char(10)
        write(u) arr
        close(u)
    end subroutine write_npy

end program geodesic
